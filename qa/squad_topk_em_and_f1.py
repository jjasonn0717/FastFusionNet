from qa.utils import _exact_match, _f1_score



def squad_topk_em_and_f1(all_topk_span_string, all_answer_strings):
    """
    This :class:`Metric` takes the topk span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score using the official SQuAD
    evaluation script.
    """
    assert len(all_topk_span_string) == len(all_answer_strings)
    all_f1 = all_em = total = 0
    for topk_span_string, answer_strings in zip(all_topk_span_string, all_answer_strings):
        if len(topk_span_string) > 0:
            exact_matches = []
            f1_scores = []
            for span_string in topk_span_string:
                exact_match = _exact_match(span_string, answer_strings)
                f1_score = _f1_score(span_string, answer_strings)
                exact_matches.append(exact_match)
                f1_scores.append(f1_score)
            all_em += max(exact_matches)
            all_f1 += max(f1_scores)
        else:
            all_em += 0
            all_f1 += 0
        total += 1
    all_em = 100. * all_em / total
    all_f1 = 100. * all_f1 / total
    return all_em, all_f1
