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


def get_pmt_batch_features(ctx, inputs):
    f1to5 = [ctx.num_model_mutants, 1, ctx.num_model_mutants, 4, 0]
    f6to10 = [1, 1, 0, 10, 0]
    f11to25 = [0, 0, 1, 0, 0]
    sample_feature = np.array(f1to5 + f6to10 + f11to25)
    batch_size = inputs.size(0)
    batch_features = np.repeat(sample_feature[None, ...], batch_size, axis=0)
    return batch_features


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
def prioritize_by_pmt(ctx):
    device = get_device(ctx)
    testloader = load_dataloader(ctx, split='test')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()


    ranker = load_pickle_object(ctx, f'pmt_ranker.{ctx.cross_model}.{ctx.feature_source}.pkl')
    assert(ranker is not None)

    mutation_score, oracle = [], []
    for inputs, targets in tqdm(testloader, desc='Batch'):
        inputs_preds = post_predict(ctx, model(inputs.to(device)))

        if ctx.feature_source == 'pmt':
            batch_features = get_pmt_batch_features(ctx, inputs)
        else:
            batch_features = get_prima_batch_features(ctx, inputs, model, device)

        ms = ranker.predict(batch_features)
        mutation_score.append(ms)

        targets = targets.to(device)
        incorrect = correctness(ctx, inputs_preds, targets, invert=True)
        oracle.append(incorrect)

    mutation_score = np.concatenate(mutation_score)
    print(mutation_score.shape)
    oracle = torch.cat(oracle).numpy()
    print(oracle.shape)

    assert(len(mutation_score) == len(oracle))
    shuffled = np.random.permutation(len(mutation_score))
    mutation_score = mutation_score[shuffled]
    oracle = oracle[shuffled]
    sortedd = mutation_score.argsort()  # ascending by default
    oracle = oracle[sortedd[::-1]]  # descending, larger ms indicates incorrect

    result = {
        'rank': oracle,
        'ideal': np.sort(oracle)[::-1],  # ascending by default, reverse it
        'worst': np.sort(oracle)  # ascending by default
    }
    save_object(ctx, result, f'pmt_list.{ctx.cross_model}.{ctx.feature_source}.pkl')


def main():
    parser.add_argument('cross_model', type=str, choices=('resnet56', 'vgg13'))
    parser.add_argument('feature_source', type=str, choices=('pmt', 'prima'))
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)
    prioritize_by_pmt(ctx)


if __name__ == '__main__':
    main()

