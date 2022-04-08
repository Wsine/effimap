import random

import torch
import torch.nn as nn
from tqdm import tqdm

from arguments import parser
from dataset import load_dataloader
from model import get_device, load_model
from extract import feature_hook, feature_container
from utils import *


@torch.no_grad()
def eval_feature_variance(model, dataloader, device):
    model.eval()

    features = []
    for inputs, targets in tqdm(dataloader, desc='Eval', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        feature_container.clear()
        model(inputs)
        features.append(torch.stack(feature_container, dim=-1))

    features = torch.cat(features)
    return features.std(dim=0)


def channel_hook(chn):
    def _hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            output[:, chn] = 0
        elif isinstance(module, nn.BatchNorm2d):
            output[:, chn] = inputs[0][:, chn]
        elif isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            try:
                d = output.get_device()
            except RuntimeError:
                d = torch.device('cpu')
            g = torch.Generator(d).manual_seed(chn)
            mask = torch.bernoulli(
                torch.where(output[0] < 0, 0.1, 0.0), generator=g
            ).bool().expand(output.size())
            output = torch.where(mask, inputs[0], output)
        return output
    return _hook


def search_model_mutants(opt, model, dataloader, device):
    hook_handlers = {}
    for name, module in model.named_modules():
        handler = module.register_forward_hook(feature_hook)
        hook_handlers[name] = handler

    var0 = eval_feature_variance(model, dataloader, device)
    var_pool = var0.clone()

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
            hook_handlers[lname].remove()
            handler = layer.register_forward_hook(channel_hook(chn_idx))  # type: ignore
            hook_handlers[lname] = layer.register_forward_hook(feature_hook)
            var = eval_feature_variance(model, dataloader, device)
            new_coverage_mask = (var - var_pool) > 0
            if new_coverage_mask.sum() > 0:
                var_pool = torch.where(new_coverage_mask, var, var_pool)
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
                    var_pool = var0.clone()
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
