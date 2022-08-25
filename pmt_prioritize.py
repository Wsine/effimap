import numpy as np
import torch
from tqdm import tqdm

from arguments import parser
from model import load_model
from dataset import load_dataloader
from metric import correctness
from utils import get_device, guard_folder, load_pickle_object, save_object


@torch.no_grad()
def prioritize_by_pmt(ctx):
    device = get_device(ctx)
    testloader = load_dataloader(ctx, split='test')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()


    ranker = load_pickle_object(ctx, f'pmt_ranker.{ctx.cross_model}.pkl')
    assert(ranker is not None)

    mutation_score, oracle = [], []
    for inputs, targets in tqdm(testloader, desc='Random'):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        f1to5 = [ctx.num_model_mutants, 1, ctx.num_model_mutants, 4, 0]
        f6to10 = [1, 1, 0, 10, 0]
        f11to25 = [0, 0, 1, 0, 0]
        sample_feature = np.array(f1to5 + f6to10 + f11to25)
        batch_size = inputs.size(0)
        batch_features = np.repeat(sample_feature[None, ...], batch_size, axis=0)
        ms = ranker.predict(batch_features)
        mutation_score.append(ms)

        incorrect = correctness(ctx, predicted, targets, invert=True)
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
    save_object(ctx, result, 'pmt_list.pkl')


def main():
    parser.add_argument('cross_model', type=str, choices=('resnet56', 'vgg13'))
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)
    prioritize_by_pmt(ctx)


if __name__ == '__main__':
    main()

