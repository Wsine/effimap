import numpy as np
import torch
from tqdm import tqdm

from arguments import parser
from model import load_model
from dataset import load_dataloader
from metric import correctness, post_predict, predicates
from generate_mutants import InverseActivate
from fuzz_mutants import SilentConv, RevertBatchNorm, RevertReLu
from utils import get_device, guard_folder, load_torch_object, save_object


@torch.no_grad()
def extract_mutant_predicates(ctx, model, dataloader, device):
    model_preds_list = []

    pred_predicates = []
    for mutant_idx in tqdm(range(ctx.num_model_mutants)):
        mutant_path = f'model_mutants/{ctx.mutant_source}_mutant.{mutant_idx}.pt'
        mutant = load_torch_object(ctx, mutant_path)
        assert(mutant is not None)
        mutant = mutant.to(device)
        mutant.eval()

        predicates_list = []
        for batch_idx, (inputs, targets) in enumerate(
                tqdm(dataloader, desc='Eval', leave=False)):
            inputs, targets = inputs.to(device), targets.to(device)

            if batch_idx > len(model_preds_list) - 1:
                model_preds = post_predict(ctx, model(inputs))
                model_preds_list.append(model_preds)
            else:
                model_preds = model_preds_list[batch_idx]

            mutant_preds = post_predict(ctx, mutant(inputs))
            pdcs = predicates(ctx, model_preds, mutant_preds)
            predicates_list.append(pdcs)
        predicates_list = torch.cat(predicates_list)

        pred_predicates.append(predicates_list)

    pred_predicates = torch.stack(pred_predicates)
    pred_predicates = pred_predicates.transpose(0, 1).numpy()

    return pred_predicates


@torch.no_grad()
def prioritize_by_comprehensive_mutation_analysis(ctx):
    device = get_device(ctx)
    testloader = load_dataloader(ctx, split='test')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()

    pred_predicates = extract_mutant_predicates(ctx, model, testloader, device)
    mutation_scores = np.sum(pred_predicates, axis=1)
    print('mutation_scores -', mutation_scores.shape)

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
    oracle = oracle[sortedd[::-1]]  # descending, larger error indicates incorrect

    result = {
        'rank': oracle,
        'ideal': np.sort(oracle)[::-1],  # ascending by default, reverse it
        'worst': np.sort(oracle)  # ascending by default
    }
    save_object(ctx, result, f'cma_list.{ctx.mutant_source}.pkl')


def main():
    parser.add_argument('mutant_source', type=str, choices=('effimap', 'random'))
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)
    prioritize_by_comprehensive_mutation_analysis(ctx)


if __name__ == '__main__':
    main()

