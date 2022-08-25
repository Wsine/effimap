def correctness(ctx, a, b, invert=False):
    """
    return: a list containing [1 if correct else 0] unless invert
    """
    if ctx.task == 'clf':
        correct = a.eq(b).int().cpu()
    else:
        raise NotImplemented

    if invert is True:
        return 1 - correct
    return correct


def post_predict(ctx, outputs):
    if ctx.task == 'clf':
        _, predicted = outputs.max(1)
    else:
        raise NotImplemented
    return predicted


def predicates(ctx, a, b):
    """
    return: a list containing [1 if inconsistent else 0]
    """
    if ctx.task == 'clf':
        pdc = a.ne(b).int().cpu()
    else:
        raise NotImplemented
    return pdc
