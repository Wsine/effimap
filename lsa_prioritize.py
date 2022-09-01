import numpy as np
import torch
from tqdm import tqdm

from arguments import parser
from model import load_model
from dataset import load_dataloader
from metric import correctness, post_predict
from train_lsa_kde import activation_hook, activation_trace
from utils import get_device, guard_folder, load_pickle_object, save_object


@torch.no_grad()
def prioritize_by_lsa(ctx):
    device = get_device(ctx)
    testloader = load_dataloader(ctx, split='test')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()

    if ctx.dataset == 'cifar100' and ctx.model == 'resnet32':
        model.layer1[4].bn1.register_forward_hook(activation_hook)

    lsa_output = load_pickle_object(ctx, 'lsa_kde_function.pkl')
    assert(lsa_output is not None)
    kdes = lsa_output['kde']
    keep_cols = lsa_output['keep_cols']

    kde_scores, oracle = [], []
    for inputs, targets in tqdm(testloader, desc='Batch'):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs_preds = post_predict(ctx, model(inputs))
        incorrect = correctness(ctx, inputs_preds, targets, invert=True)
        oracle.append(incorrect)

        inputs_labels, _ = inputs_preds
        ats = activation_trace[-1][:, keep_cols]
        # scores = kde.score_samples(ats)
        # kde_scores.append(scores)
        for at, pred_c in zip(ats, inputs_labels):
            score = np.array(-1. * kdes[pred_c.item()].logpdf(at)).item()
            kde_scores.append(score)

    kde_scores = np.concatenate(kde_scores) * -1.
    print(kde_scores.shape)
    oracle = torch.cat(oracle).numpy()
    print(oracle.shape)

    assert(len(kde_scores) == len(oracle))
    sortedd = kde_scores.argsort()  # ascending by default
    oracle = oracle[sortedd]  # descending, smaller score indicates incorrect

    result = {
        'rank': oracle,
        'ideal': np.sort(oracle)[::-1],  # ascending by default, reverse it
        'worst': np.sort(oracle)  # ascending by default
    }
    save_object(ctx, result, f'lsa_list.pkl')


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)
    prioritize_by_lsa(ctx)


if __name__ == '__main__':
    main()

