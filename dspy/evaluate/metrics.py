# TODO: This should move internally. Same for passage_match. dspy.metrics.answer_exact_match, dspy.metrics.answer_passage_match


from dsp.primitives.demonstrate import answer_match, passage_match


def answer_exact_match(example, pred, trace=None, frac=1.0):
    assert type(example.answer) is str or type(example.answer) is list

    if type(example.answer) is str:
        return answer_match(pred.answer, [example.answer], frac=frac)
    else:  # type(example.answer) is list
        return answer_match(pred.answer, example.answer, frac=frac)


answer_exact_match_str = answer_match


def answer_passage_match(example, pred, trace=None):
    assert type(example.answer) is str or type(example.answer) is list

    if type(example.answer) is str:
        return passage_match(pred.context, [example.answer])
    else:  # type(example.answer) is list
        return passage_match(pred.context, example.answer)
