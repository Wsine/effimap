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
