import json
import numpy as np
import torch
import math
import time
import spacy
import collections

from qa.utils import load_data
from qa.model_ans_extraction import DocReaderModel
from qa.general_utils import pre_proc_fusion as pre_proc
from qa.general_utils import normalize_text


def split_batch_dict(d):
    batch_size = len(d[list(d.keys())[0]])
    assert all([len(v) == batch_size for k, v in d.items()])
    split_dicts = [{k: None for k in d} for _ in range(batch_size)]
    for k in d:
        for i in range(batch_size):
            split_dicts[i][k] = d[k][i]
    return split_dicts


def get_spans(tokens, text):
    pos = 0
    spans = []
    for token in tokens:
        start = pos + text[pos:].find(token)
        spans.append([start, start+len(token)])
        pos = start + len(token)
    for (s, e), token in zip(spans, tokens):
        assert text[s:e] == token, '{}, {}\ntext: {}\n token: {}'.foramt(s, e, text, token)
    return spans


def token2id(doc, vocab, unk_id=None):
    w2id = vocab['w2id']
    ids = [w2id[w] if w in w2id else unk_id for w in doc]
    return ids


def filter_spans(topk_span, topk_logprob, topk_span_str):
    res_spans = []
    res_logprobs = []
    for sp_i, (sp, logprob) in enumerate(zip(topk_span, topk_logprob)):
        if sp[0] == -1:
            assert sp[1] == -1, sp
            assert math.isclose(logprob, -1e30, rel_tol=1e-6), logprob
            break
        res_spans.append(sp)
        res_logprobs.append(logprob)
    assert all(s == -1 and e == -1 for s, e in topk_span[sp_i:]) or (sp_i == len(topk_span)-1 and not sp[0] == -1 and not sp[1] == -1), topk_span
    assert len(res_spans) == len(topk_span_str)
    return res_spans, res_logprobs


def span_overlap(sp1, sp2):
    s1, e1 = sp1
    s2, e2 = sp2
    if s1 <= s2 <= e1:
        return True
    elif s1 <= e2 <= e1:
        return True
    elif s2 <= s1 <= e2:
        assert s2 <= e1 <= e2
        return True
    else:
        return False

def get_nonoverlap_span(topk_span, topk_logprob):
    res_spans = []
    res_logprob = []
    res_spans_idx = []
    for sp_i, (sp, logprob) in enumerate(zip(topk_span, topk_logprob)):
        if any(span_overlap(sp, prev_sp) for prev_sp in res_spans):
            continue
        res_spans.append(sp)
        res_logprob.append(logprob)
        res_spans_idx.append(sp_i)
    return res_spans, res_logprob, res_spans_idx


def get_span_logit(span, start_logits, end_logits):
    s, e = span
    return start_logits[s] + end_logits[e]


class FusionNetPredictor:
    def __init__(self, model_path, data_path, ans_top_k=None, cuda=True):
        print('[loading previous model...]')
        checkpoint = torch.load(model_path)
        # reset ans_top_k
        if ans_top_k is None:
            ans_top_k = checkpoint['config']['ans_top_k'] if 'ans_top_k' in checkpoint['config'] else 20
        args = {'seed': 123,
                'cuda': cuda,
                'debug': False,
                'max_eval_len': 0,
                'ans_top_k': ans_top_k
                }
        checkpoint['config'].update(args)
        opt = checkpoint['config']
        print("ans_top_k:", opt['ans_top_k'])
        _, _, _, _, embedding, opt, meta = load_data(opt, log=None, data_path=data_path)

        self._eval_vocab = {'vocab': meta['vocab'], 'w2id': {w: i for i, w in enumerate(meta['vocab'])}}
        self._vocab_tag = {'vocab': meta['vocab_tag'], 'w2id': {w: i for i, w in enumerate(meta['vocab_tag'])}}
        self._vocab_ent = {'vocab': meta['vocab_ent'], 'w2id': {w: i for i, w in enumerate(meta['vocab_ent'])}}

        # load model
        state_dict = checkpoint['state_dict']
        self._model = DocReaderModel(opt, embedding, state_dict)
        if cuda:
            self._model.cuda()

        # preprocess nlp
        self._nlp = spacy.load('en', parser=False)

        self._opt = {'tf': True, 'use_feat_emb': True, 'pos_size': 12, 'ner_size': 8, 'use_elmo': False}
        self._opt.update(opt)

    def _json_to_instance(self, article):
        article_id = article['_id']
        answers = article['answers']

        question = article['question'].strip().replace("\n", "")
        question_text = pre_proc(question)
        question_doc = self._nlp(question_text)
        question_tokens = [normalize_text(w.text) for w in question_doc]
        question_token_span = get_spans([w.text for w in question_doc], question)
        question_ids = token2id(question_tokens, self._eval_vocab, unk_id=1)

        start = time.time()
        context = ""
        context_tokens = []
        sents_docs = []
        token_spans_sent = []
        paragraphs = article['context']
        for para in paragraphs:
            cur_title, cur_para = para[0], para[1]
            for sent_id, sent in enumerate(cur_para):
                sent_text = pre_proc(sent)
                sent_doc = self._nlp(sent_text)
                sent_tokens = [normalize_text(w.text) for w in sent_doc]
                if len(sent_tokens) == 0:
                    continue
                sents_docs.append(sent_doc)
                token_spans_sent.append([len(context_tokens), len(context_tokens)+len(sent_tokens)-1])
                context += sent
                context_tokens.extend(sent_tokens)
        time_doc_preprocess = time.time() - start
        #print("doc preprocess:", time_doc_preprocess)

        context_token_span = get_spans([w.text for doc in sents_docs for w in doc], context)
        context_sentence_lens = [[] for doc in sents_docs]

        # get features
        context_tags = [w.tag_ for doc in sents_docs for w in doc]
        context_ents = [w.ent_type_ for doc in sents_docs for w in doc]
        question_word = {w.text for w in question_doc}
        question_lower = {w.text.lower() for w in question_doc}
        question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question_doc}
        match_origin = [w.text in question_word for doc in sents_docs for w in doc]
        match_lower = [w.text.lower() in question_lower for doc in sents_docs for w in doc]
        match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for doc in sents_docs for w in doc]
        context_features = list(zip(match_origin, match_lower, match_lemma))

        context_ids = token2id(context_tokens, self._eval_vocab, unk_id=1)

        # term frequency in document
        counter_ = collections.Counter(w.lower() for w in context_tokens)
        total = sum(counter_.values())
        context_tf = [counter_[w.lower()] / total for w in context_tokens]
        context_features = [list(w) + [tf] for w, tf in zip(context_features, context_tf)]

        context_tag_ids = token2id(context_tags, self._vocab_tag)

        context_ent_ids = token2id(context_ents, self._vocab_ent)

        instance = (article_id,
                    context_ids,
                    context_features,
                    context_tag_ids,
                    context_ent_ids,
                    question_ids,
                    context,
                    context_token_span,
                    context_sentence_lens,
                    None,
                    None,
                    question,
                    question_token_span,
                    answers,
                    )
        metadata = {'token_spans_sent': token_spans_sent,
                    'passage_tokens': context_tokens,
                    'question_tokens': question_tokens,
                    'answers': answers,
                    '_id': article_id,
                    'time_doc_preprocess': time_doc_preprocess}
        return instance, metadata

    def _batchify(self, batch):
        batch_size = len(batch)
        batch = list(zip(*batch))

        with torch.no_grad():
            context_len = max(len(x) for x in batch[1])
            # print('context_len:', context_len)
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[2][0][0])

            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)
            if not self._opt['tf']:
                if self._opt['match']:
                    context_feature = context_feature[:, :, :3]
                else:
                    context_feature = None
            else:
                if not self._opt['match']:
                    context_feature = context_feature[:, :, 3:]

            if self._opt['use_feat_emb']:
                context_tag = torch.LongTensor(batch_size, context_len).fill_(0)
                for i, doc in enumerate(batch[3]):
                    context_tag[i, :len(doc)] = torch.LongTensor(doc)

                context_ent = torch.LongTensor(batch_size, context_len).fill_(0)
                for i, doc in enumerate(batch[4]):
                    context_ent[i, :len(doc)] = torch.LongTensor(doc)
            else:
                # create one-hot vectors
                context_tag = torch.Tensor(batch_size, context_len, self._opt['pos_size']).fill_(0)
                for i, doc in enumerate(batch[3]):
                    for j, tag in enumerate(doc):
                        context_tag[i, j, tag] = 1

                context_ent = torch.Tensor(batch_size, context_len, self._opt['ner_size']).fill_(0)
                for i, doc in enumerate(batch[4]):
                    for j, ent in enumerate(doc):
                        context_ent[i, j, ent] = 1

            question_len = max(len(x) for x in batch[5])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            text = list(batch[6])
            span = list(batch[7])
            context_sentence_lens = list(batch[8])

            context_char_id, question_char_id = None, None

            if self._opt['cuda']:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory() if context_feature is not None else None
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
                context_char_id = context_char_id.cuda() if context_char_id is not None else None
                question_char_id = question_char_id.cuda() if question_char_id is not None else None
        return (context_id, context_feature, context_tag, context_ent, context_mask,
                question_id, question_mask, context_sentence_lens, context_char_id, question_char_id, text, span)

    def process_output(self, output, metadata):
        """
        output_dict = {
            "span_start_logits",
            "span_end_logits",
            "best_span",
            "topk_span",
            "topk_logprob",
            "best_span_str",
            "topk_span_str"}
        """
        topk_span_str = output['topk_span_str']
        best_span_str = output['best_span_str']
        topk_span = output['topk_span']
        topk_logprob = output['topk_logprob']
        start_logits = output['span_start_logits']
        end_logits = output['span_end_logits']
        topk_span, topk_logprob = filter_spans(topk_span, topk_logprob, topk_span_str)
        nonoverlap_spans, nonoverlap_logprob, nonoverlap_spans_idx = get_nonoverlap_span(topk_span, topk_logprob)
        nonoverlap_span_strs = [topk_span_str[i] for i in nonoverlap_spans_idx]
        return {'answer_texts': metadata['answers'],
                'best_span_str': output.get('best_span_str', None),
                'topk_span_str': output.get('topk_span_str', None),
                #'best_span': output.get('best_span', None),
                'topk_span': topk_span,
                'topk_logprob': topk_logprob,
                'topk_logit': [get_span_logit(sp, start_logits, end_logits) for sp in topk_span],
                'nonoverlap_span': nonoverlap_spans,
                'nonoverlap_span_str': nonoverlap_span_strs,
                'nonoverlap_logprob': nonoverlap_logprob,
                'nonoverlap_logit': [get_span_logit(sp, start_logits, end_logits) for sp in nonoverlap_spans],
                'question_tokens': metadata['question_tokens'],
                'passage_tokens': metadata['passage_tokens'],
                'token_spans_sent': metadata['token_spans_sent'],
                "start_logits": start_logits,
                "end_logits": end_logits,
                '_id': metadata['_id'],
                'time_doc_preprocess': metadata['time_doc_preprocess']}

    def predict_json(self, article):
        instance, metadata = self._json_to_instance(article)
        batch = self._batchify([instance])
        output = split_batch_dict(self._model.predict(batch, full_info=True))[0]
        return self.process_output(output, metadata)

    def predict_batch_json(self, articles):
        instances, metadatas = list(zip(*[self._json_to_instance(article) for article in articles]))
        batch = self._batchify(instances)
        outputs = split_batch_dict(self._model.predict(batch, full_info=True))
        return [self.process_output(o, m) for o, m in zip(outputs, metadatas)]

    def predict(self, hotpot_dict_instances):
        """
        Expects JSON that has the same format of instances in Hotpot dataset
        """
        assert type(hotpot_dict_instances) == list
        if len(hotpot_dict_instances) == 1:
            return [self.predict_json(hotpot_dict_instances[0])]
        else:
            return self.predict_batch_json(hotpot_dict_instances)
