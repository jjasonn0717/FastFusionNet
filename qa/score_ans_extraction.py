from qa.utils import score
from qa.topk_mention_recall import topk_mention_recall
from qa.squad_topk_em_and_f1 import squad_topk_em_and_f1


def score_ans_extraction(pred, truth):
    em, f1 = score([d['best_span_str'] for d in pred], truth)
    topk_em, topk_f1 = squad_topk_em_and_f1([d['topk_span_str'] for d in pred], truth)
    topk_recall = topk_mention_recall([d['topk_span_str'] for d in pred], truth)
    return em, f1, topk_em, topk_f1, topk_recall
