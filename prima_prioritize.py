import numpy as np
import torch
from tqdm import tqdm

from arguments import parser
from model import load_model
from dataset import load_dataloader
from metric import correctness, post_predict
from generate_mutants import InverseActivate
from train_prima_ranker import compute_prima_samples_features
from train_prima_ranker import extract_prima_model_mutants_predictions
from train_prima_ranker import extract_prima_sample_mutants_predictions
from utils import get_device, guard_folder, load_pickle_object, save_object


@torch.no_grad()
def extract_prima_sample_features(ctx, model, valloader, device):
    model_mutants_preds = \
        extract_prima_model_mutants_predictions(ctx, valloader, device)

    sample_mutants_preds = \
        extract_prima_sample_mutants_predictions(ctx, model, valloader, device)

    sample_features = []
    if ctx.task == 'clf':
        model_mutants_labels, model_mutants_probs = model_mutants_preds
        sample_mutants_labels, sample_mutants_probs = sample_mutants_preds
        for mm_labels, mm_probs, sm_labels, sm_probs in zip(
                model_mutants_labels, model_mutants_probs,
                sample_mutants_labels, sample_mutants_probs):
            input_pred = (sm_labels[0], sm_probs[0])
            samples_pred = (sm_labels[1:], sm_probs[1:])
            models_pred = (mm_labels, mm_probs)
            feature = compute_prima_samples_features(
                ctx, input_pred, samples_pred, models_pred)
            sample_features.append(feature)
    else:
        raise NotImplemented

    sample_features = torch.stack(sample_features).numpy()
    return sample_features


@torch.no_grad()
def prioritize_by_prima(ctx):
    device = get_device(ctx)
    testloader = load_dataloader(ctx, split='test')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()

    sample_features = extract_prima_sample_features(ctx, model, testloader, device)

    ranker = load_pickle_object(ctx, f'prima_ranker.pkl')
    assert(ranker is not None)

    model_errors = ranker.predict(sample_features)
    print(model_errors.shape)

    oracle = []
    for inputs, targets in tqdm(testloader, desc='Batch'):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_preds = post_predict(ctx, model(inputs))
        incorrect = correctness(ctx, inputs_preds, targets, invert=True)
        oracle.append(incorrect)

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

