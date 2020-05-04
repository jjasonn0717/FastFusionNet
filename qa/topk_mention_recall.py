from qa.utils import _normalize_answer


def topk_mention_recall(all_topk_predictions, all_answer_texts, lower=False, normalize=False, all_ans=True):
    assert len(all_topk_predictions) == len(all_answer_texts)
    num_gold_mentions = 0
    num_recalled_mentions = 0

    for topk_predictions, answer_texts in zip(all_topk_predictions, all_answer_texts):
        if lower:
            topk_predictions = [t.lower() for t in topk_predictions]
            answer_texts = [t.lower() for t in answer_texts]
        if normalize:
            topk_predictions = [_normalize_answer(t) for t in topk_predictions]
            answer_texts = [_normalize_answer(t) for t in answer_texts]
        topk_predictions = set(topk_predictions)
        answer_texts = set(answer_texts)
        if all_ans:
            num_gold_mentions += len(answer_texts)
            num_recalled_mentions += len(answer_texts & topk_predictions)
        else:
            num_gold_mentions += 1
            num_recalled_mentions += int(len(answer_texts & topk_predictions) > 0)

    if num_gold_mentions == 0:
        recall = 0.0
    else:
        recall = num_recalled_mentions/float(num_gold_mentions)
    return recall
