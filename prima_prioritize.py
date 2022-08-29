import numpy as np
import torch
from tqdm import tqdm

from arguments import parser
from model import load_model
from dataset import load_dataloader
from metric import correctness, post_predict
from generate_mutants import generate_random_sample_mutants
from train_pmt_ranker import compute_prima_sample_feature
from utils import get_device, guard_folder, load_pickle_object, save_object


def get_prima_batch_features(ctx, inputs, model, device):
    sample_features = []
    for sample_with_mutants in generate_random_sample_mutants(ctx, inputs):
        inputs = sample_with_mutants.to(device)
        inputs_preds = post_predict(ctx, model(inputs))
        feature = compute_prima_sample_feature(ctx, inputs_preds)
        feature = feature.cpu()
        sample_features.append(feature)
    sample_features = torch.stack(sample_features).numpy()

    return sample_features


@torch.no_grad()
def prioritize_by_prima(ctx):
    device = get_device(ctx)
    testloader = load_dataloader(ctx, split='test')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()


    ranker = load_pickle_object(ctx, f'prima_ranker.pkl')
    assert(ranker is not None)

    model_errors, oracle = [], []
    for inputs, targets in tqdm(testloader, desc='Batch'):
        inputs_preds = post_predict(ctx, model(inputs.to(device)))

        batch_features = get_prima_batch_features(ctx, inputs, model, device)

        pred_errors = ranker.predict(batch_features)
        model_errors.append(pred_errors)

        targets = targets.to(device)
        incorrect = correctness(ctx, inputs_preds, targets, invert=True)
        oracle.append(incorrect)

    model_errors = np.concatenate(model_errors)
    print(model_errors.shape)
    oracle = torch.cat(oracle).numpy()
    print(oracle.shape)

    assert(len(model_errors) == len(oracle))
    shuffled = np.random.permutation(len(model_errors))
    model_errors = model_errors[shuffled]
    oracle = oracle[shuffled]
    sortedd = model_errors.argsort()  # ascending by default
    oracle = oracle[sortedd[::-1]]  # descending, larger error indicates incorrect

    result = {
        'rank': oracle,
        'ideal': np.sort(oracle)[::-1],  # ascending by default, reverse it
        'worst': np.sort(oracle)  # ascending by default
    }
    save_object(ctx, result, f'prima_list.pkl')


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)
    prioritize_by_prima(ctx)


if __name__ == '__main__':
    main()

