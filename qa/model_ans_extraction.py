# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa
#
# Modified by Felix Wu
# Modification:
#   - change the logger name
#   - save & load optimizer state dict
#   - change the dimension of inputs (for POS and NER features)

import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from heapq import heappush, heappop, heappushpop, heapify

from .utils import AverageMeter, EMA
from .rnn_reader import *


logger = logging.getLogger(__name__)


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, embedding=None, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self._ans_top_k = opt['ans_top_k']
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()

        # Building network.
        if opt['model_type'] == 'drqa':
            self.network = RnnDocReader(opt, embedding=embedding)
        elif opt['model_type'] == 'gldr-drqa':
            self.network = CnnDocReader(opt, embedding=embedding)
        elif opt['model_type'] == 'fusionnet':
            self.network = FusionNet(opt, embedding=embedding)
        elif opt['model_type'] == 'bidaf':
            self.network = BiDAF(opt, embedding=embedding)
        else:
            print('UNKNOWN model_type: ' + opt['model_type'])
            raise NotImplementedError
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, opt['learning_rate'],
                                          weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, opt['learning_rate'],
                                        betas=(opt['beta1'], opt['beta2']),
                                        weight_decay=opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        print(self.optimizer)

        if opt['ema_decay'] < 1.:
            print('using EMA')
            self.ema = EMA(opt['ema_decay'])
            for name, param in self.network.named_parameters():
                if param.requires_grad:
                    self.ema.register(name, param.data)

        num_params = sum(p.data.numel() for p in parameters
            if p.data.data_ptr() != self.network.embedding.weight.data.data_ptr())
        print("{} parameters".format(num_params))

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        with torch.no_grad():
            if self.opt['cuda']:
                inputs = [e.cuda(non_blocking=True) if torch.is_tensor(e) else e for e in ex[:10]]
                target_s, target_e = ex[10].cuda(non_blocking=True), ex[11].cuda(non_blocking=True)
            else:
                inputs = ex[:10]
                target_s, target_e = ex[10], ex[11]

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        self.train_loss.update(loss.item(), ex[0].size(0))

        # warm_start
        if self.opt['warm_start'] and self.updates <= 1000:
            lr = self.opt['learning_rate'] / math.log(1002.) * math.log(self.updates + 2)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if self.opt['grad_clipping'] > 0.:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                           self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Exponential Moving Average
        if hasattr(self, 'ema'):
            for name, param in self.network.named_parameters():
               if param.requires_grad:
                   param.data = self.ema(name, param.data)

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict(self, ex, full_info=False):
        # Eval mode
        self.network.eval()

        with torch.no_grad():
            # Transfer to GPU
            if next(self.network.parameters()).is_cuda:
                inputs = [e.cuda(non_blocking=True) if torch.is_tensor(e) else e for e in ex[:10]]
            else:
                inputs = ex[:10]

            if not full_info:
                # Run forward
                score_s, score_e = self.network(*inputs)
                if type(score_s) is list:
                    score_s, score_e = score_s[-1], score_e[-1]


                # Transfer to CPU/normal tensors for numpy ops
                score_s = score_s.data.cpu()
                score_e = score_e.data.cpu()

                # Get argmax text spans
                text = ex[-2]
                spans = ex[-1]
                predictions = []
                max_len = self.opt['max_len'] or score_s.size(1)
                for i in range(score_s.size(0)):
                    scores = torch.ger(score_s[i], score_e[i])
                    scores.triu_().tril_(max_len - 1)
                    scores = scores.numpy()
                    s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
                    s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                    predictions.append(text[i][s_offset:e_offset])
                return predictions
            else:
                # Run forward
                score_s, score_e = self.network(*inputs, logit=True)
                if type(score_s) is list:
                    score_s, score_e = score_s[-1], score_e[-1]

                # get topk span
                span_start_logprobs = F.log_softmax(score_s, 1)
                span_end_logprobs = F.log_softmax(score_e, 1)

                passage_str = ex[-2]
                offsets = ex[-1]
                metadata = {'original_passage': passage_str, 'token_offsets': offsets}
                topk_span, topk_logprob = self.get_best_span(span_start_logprobs, span_end_logprobs, self._ans_top_k, metadata)
                output_dict = {
                    "span_start_logits": score_s.detach().cpu().numpy(),
                    "span_end_logits": score_e.detach().cpu().numpy(),
                    "best_span": topk_span[:, 0, :].detach().cpu().numpy(),
                    "topk_span": topk_span.detach().cpu().numpy(),
                    "topk_logprob": topk_logprob.detach().cpu().numpy(),
                    'best_span_str': [],
                    'topk_span_str': [],
                }
                for i in range(score_s.size(0)):
                    topk_predicted_span = topk_span[i].detach().cpu().numpy()
                    topk_span_string = []
                    for k in range(self._ans_top_k):
                        if topk_predicted_span[k, 0] == -1:
                            continue
                        start_offset = offsets[i][topk_predicted_span[k, 0]][0]
                        end_offset = offsets[i][topk_predicted_span[k, 1]][1]
                        topk_span_string.append(passage_str[i][start_offset:end_offset])
                    output_dict['best_span_str'].append(topk_span_string[0])
                    output_dict['topk_span_str'].append(topk_span_string)
                return output_dict

    @staticmethod
    def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor, ans_top_k: int, meta) -> torch.Tensor:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        #batch_size, passage_length = span_start_logits.size()
        batch_size, batch_passage_length = span_start_logits.size()
        topk_word_span = span_start_logits.new_zeros((batch_size, ans_top_k, 2), dtype=torch.long) - 1
        topk_logprob = span_start_logits.new_zeros((batch_size, ans_top_k)) - 1e30

        span_start_logits = span_start_logits.detach().cpu().numpy()
        span_end_logits = span_end_logits.detach().cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            passage_str = meta['original_passage'][b]
            offsets = meta['token_offsets'][b]
            passage_length = min(len(offsets), batch_passage_length)
            span_log_prob_topk = []
            for i in range(passage_length):
                end = min(i+10, passage_length)
                val1 = span_start_logits[b, i]
                for j in range(i, end):
                    val2 = span_end_logits[b, j]
                    if len(span_log_prob_topk) == ans_top_k:
                        heappushpop(span_log_prob_topk, (val1 + val2, i, j))
                    else:
                        heappush(span_log_prob_topk, (val1 + val2, i, j))
            actual_cand_num = min(ans_top_k, len(span_log_prob_topk))
            for k in range(actual_cand_num):
                logprob, s_idx, e_idx = heappop(span_log_prob_topk)
                topk_word_span[b, actual_cand_num - k - 1, 0] = s_idx
                topk_word_span[b, actual_cand_num - k - 1, 1] = e_idx
                topk_logprob[b, actual_cand_num - k - 1] = logprob.astype(np.float)
                '''
                if b == 0:
                    print(logprob)
                '''
            assert len(span_log_prob_topk) == 0, repr(span_log_prob_topk)
        return topk_word_span, topk_logprob

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename, epoch, best_val_score=0.):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch,
            'best_val_score': best_val_score,
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
        return self

    def cpu(self):
        self.network.cpu()
        return self
