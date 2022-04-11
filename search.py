import random

import torch
import torch.nn as nn
from tqdm import tqdm

from arguments import parser
from dataset import load_dataloader
from model import get_device, load_model
from mutate import channel_hook
from utils import *


@torch.no_grad()
def eval_performance(model, dataloader, device):
    model.eval()

    preds, gt = [], []
    for inputs, targets in tqdm(dataloader, desc='Eval', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        preds.append(predicted)
        gt.append(predicted.eq(targets))

    preds = torch.cat(preds)
    gt = torch.cat(gt)
    return preds, gt


def search_model_mutants(opt, model, dataloader, device):
    pred0, gt0 = eval_performance(model, dataloader, device)
    mutation_pool = torch.zeros_like(gt0, dtype=torch.bool)
    step_mutation_score = torch.zeros(1).to(device)

    result = {}
    searched = [f"{v['layer_name']}_{v['channel_idx']}" for v in result.values()]
    fixed_random = random.Random(opt.seed)
    valid_layers = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU))
    ]
    mutant_idx = len(result.keys()) + 1
    with tqdm(total=opt.num_model_mutants, desc='Mutants') as pbar:
        accu_trial = 0
        while mutant_idx <= opt.num_model_mutants:
            lname, layer, chn_idx = None, None, None
            while layer is None:
                lname, layer = fixed_random.choice(valid_layers)
                if isinstance(layer, nn.Conv2d):
                    chn_idx = fixed_random.choice(range(layer.out_channels))
                elif isinstance(layer, nn.BatchNorm2d):
                    chn_idx = fixed_random.choice(range(layer.num_features))
                elif isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                    chn_idx = opt.seed + mutant_idx
                else:
                    raise ValueError('Impossible')
                if f'{lname}_{chn_idx}' in searched:
                    layer = None
            handler = layer.register_forward_hook(channel_hook(chn_idx))  # type: ignore
            pred, _ = eval_performance(model, dataloader, device)
            mutated_mask = pred.ne(pred0)
            mutated_score = torch.logical_and(mutated_mask, ~gt0).sum() \
                          - torch.logical_and(~mutated_mask, gt0).sum()
            new_coverage_mask = ~gt0 & mutated_mask & ~mutation_pool
            if new_coverage_mask.sum() > 0 or mutated_score > step_mutation_score:
                mutation_pool.logical_or_(new_coverage_mask)
                step_mutation_score = step_mutation_score  \
                    + 0.5 * (mutated_score - step_mutation_score)
                result[f'mutant{mutant_idx}'] = {
                    'layer_name': lname,
                    'channel_idx': chn_idx
                }
                searched.append(f'{lname}_{chn_idx}')
                mutant_idx += 1
                pbar.update(1)
                accu_trial = 0
            else:
                accu_trial += 1
                if accu_trial > opt.num_model_mutants:
                    print(f'Detected no new coverage after {opt.num_model_mutants} trials')
                    mutation_pool.zero_()
                    step_mutation_score.zero_()
                    accu_trial = 0
            handler.remove()

    return result, 'model_mutants_info.json'


def main():
    opt = parser.parse_args()
    print(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    valloader = load_dataloader(opt, split='val')
    results = search_model_mutants(opt, model, valloader, device)
    save_object(opt, *results)


if __name__ == '__main__':
    main()
