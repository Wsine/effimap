import torch.nn.functional as F


def correctness(ctx, preds, targets, invert=False):
    """
    return: a list containing [1 if correct else 0] unless invert
    """
    if ctx.task == 'clf':
        labels, _ = preds
        correct = labels.eq(targets).int().cpu()
    else:
        raise NotImplemented

    if invert is True:
        return 1 - correct
    return correct


def post_predict(ctx, outputs):
    if ctx.task == 'clf':
        _, labels = outputs.max(1)
        probs = F.softmax(outputs, dim=1)
        return (labels, probs)
    else:
        raise NotImplemented


def predicates(ctx, model_preds, mutant_preds):
    """
    return: a list containing [1 if inconsistent else 0]
    """
    if ctx.task == 'clf':
        model_labels, _ = model_preds
        mutant_labels, _ = mutant_preds
        pdc = model_labels.ne(mutant_labels).int().cpu()
    else:
        raise NotImplemented
    return pdc


def prediction_error(ctx, preds, targets):
    if ctx.task == 'clf':
        labels, _ = preds
        return labels.ne(targets).int().cpu()
    else:
        raise NotImplemented

