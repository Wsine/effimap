import random
import time

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
    acc = gt.sum().item() / gt.size(0)
    return preds, gt, acc


def search_model_mutants(opt, model, dataloader, device):
    pred0, gt0, acc0 = eval_performance(model, dataloader, device)
    mutated_pool = torch.zeros_like(gt0, dtype=torch.bool)
    minimal_mutant_score, minimal_idx = 10e9, None

    result = {}
    searched = [f"{v['layer_name']}_{v['channel_idx']}" for v in result.values()]
    fixed_random = random.Random(opt.seed)
    valid_layers = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU))
    ]
    mutant_idx = len(result.keys()) + 1
    with tqdm(total=opt.num_model_mutants, desc='Mutants') as pbar:
        start_time = time.time()
        while True:
            current_time = time.time()
            if (current_time - start_time) / 60 > opt.fuzz_energy:
                break

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

            searched.append(f'{lname}_{chn_idx}')
            handler = layer.register_forward_hook(channel_hook(chn_idx))  # type: ignore
            pred, _, acc = eval_performance(model, dataloader, device)
            if abs(acc - acc0) > 0.05:
                handler.remove()
                continue

            mutated_mask = pred.ne(pred0)
            mutant_score = torch.logical_and(mutated_mask, ~gt0).sum() \
                         - torch.logical_and(mutated_mask, gt0).sum()
            new_coverage_mask = ~gt0 & mutated_mask & ~mutated_pool

            if len(result.keys()) < opt.num_model_mutants:
                mutated_pool.logical_or_(new_coverage_mask)
                result[f'mutant{mutant_idx}'] = {
                    'layer_name': lname,
                    'channel_idx': chn_idx,
                    'mutant_score': mutant_score.item(),
                    'new_coverage': new_coverage_mask.sum().item() > 0
                }
                mutant_idx += 1
                pbar.update(1)
                if len(result.keys()) == opt.num_model_mutants:
                    for k, v in result.items():
                        if v['mutant_score'] < minimal_mutant_score:
                            minimal_mutant_score = v['mutant_score']
                            minimal_idx = k[len('mutant'):]
                handler.remove()
                continue
            elif new_coverage_mask.sum() > 0:
                for k, v in result.items():
                    if v['new_coverage'] is False:
                        result[k] = {
                            'layer_name': lname,
                            'channel_idx': chn_idx,
                            'mutant_score': mutant_score.item(),
                            'new_coverage': True
                        }
                        break
            elif mutant_score > minimal_mutant_score:
                result[f'mutant{minimal_idx}'] = {
                    'layer_name': lname,
                    'channel_idx': chn_idx,
                    'mutant_score': mutant_score.item(),
                    'new_coverage': False
                }
                for k, v in result.items():
                    if v['mutant_score'] < minimal_mutant_score:
                        minimal_mutant_score = v['mutant_score']
                        minimal_idx = k[len('mutant'):]
            handler.remove()

    return result, 'model_mutants_info.json'


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    valloader = load_dataloader(opt, split='val')
    results = search_model_mutants(opt, model, valloader, device)
    save_object(opt, *results)


if __name__ == '__main__':
    main()
