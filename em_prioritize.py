import numpy as np
import torch
from tqdm import tqdm

from arguments import parser
from model import load_model
from dataset import load_dataloader
from metric import correctness, post_predict
from train_em_estimator import extract_effimap_sample_features
from utils import get_device, guard_folder, load_pickle_object, save_object


@torch.no_grad()
def prioritize_by_effimap(ctx):
    device = get_device(ctx)
    testloader = load_dataloader(ctx, split='test')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()

    sample_features = extract_effimap_sample_features(ctx, model, testloader, device)

    ranker = load_pickle_object(ctx, f'effimap_ranker.pkl')
    assert(ranker is not None)

    mutant_predicates = ranker.predict(sample_features)
    print(mutant_predicates.shape)
    # mutant_predicates = np.where(mutant_predicates > 0.5, 1, 0)
    mutation_scores = np.sum(mutant_predicates, axis=1)
    print(mutation_scores.shape)

    oracle = []
    for inputs, targets in tqdm(testloader, desc='Batch'):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_preds = post_predict(ctx, model(inputs))
        incorrect = correctness(ctx, inputs_preds, targets, invert=True)
        oracle.append(incorrect)

    oracle = torch.cat(oracle).numpy()
    print(oracle.shape)

    assert(len(mutation_scores) == len(oracle))
    sortedd = mutation_scores.argsort()  # ascending by default
    oracle = oracle[sortedd[::-1]]  # descending, larger score indicates incorrect

    result = {
        'rank': oracle,
        'ideal': np.sort(oracle)[::-1],  # ascending by default, reverse it
        'worst': np.sort(oracle)  # ascending by default
    }
    save_object(ctx, result, f'effimap_list.pkl')


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)
    prioritize_by_effimap(ctx)


if __name__ == '__main__':
    main()

