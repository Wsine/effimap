import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_dataloader
from model import get_device, load_model
from arguments import parser
from utils import *


def get_tensor_mask(tensor, seed, ratio=0.1):
    g = torch.Generator().manual_seed(seed)
    num_neurons = tensor.numel()
    num_mask = int(num_neurons * ratio)
    shuffle = torch.randperm(num_neurons, generator=g)
    mask = torch.cat(
        (torch.ones((num_mask,), dtype=torch.long),
         torch.zeros((num_neurons - num_mask,), dtype=torch.long))
    )[shuffle].reshape_as(tensor).bool()
    return mask


def NAI_hook(rand_seed):
    def _hook(module, inputs):
        mask = get_tensor_mask(inputs[0][0], rand_seed)
        try:
            device = inputs[0].get_device()
            mask = mask.cuda(device)
        except RuntimeError:
            pass
        for batch_input in inputs:  # tuple
            for single_input in batch_input:
                single_input = torch.where(mask, single_input.mul(-1.), single_input)
        return inputs
    return _hook


def NEB_hack(tensor, rand_seed, fill=0, **kwargs):
    mask = get_tensor_mask(tensor, rand_seed, **kwargs)
    tensor.masked_fill_(mask, fill)
    return tensor


def GF_hack(tensor, rand_seed, std=0.1, **kwargs):
    mask = get_tensor_mask(tensor, rand_seed, **kwargs)
    g = torch.Generator().manual_seed(rand_seed)
    gauss = torch.normal(0, std, tensor.size(), generator=g)
    if isinstance(tensor, nn.parameter.Parameter):
        tensor.copy_(torch.where(mask, tensor + gauss, tensor))
    else:
        tensor = torch.where(mask, tensor + gauss, tensor)
    return tensor


def WS_hack(tensor, rand_seed, ratio=0.1):
    mask = get_tensor_mask(tensor, rand_seed, ratio)
    mask_indices = mask.nonzero(as_tuple=True)
    g = torch.Generator().manual_seed(rand_seed)
    shuffle = torch.randperm(mask_indices[0].size(0), generator=g)
    tensor[mask_indices] = tensor[mask_indices][shuffle]
    return tensor


def PCR_hack(tensor, rand_seed, **kwargs):
    mask = get_tensor_mask(tensor, rand_seed, **kwargs)
    tensor = torch.where(mask, 1 - tensor, tensor)
    return tensor


def get_model_mutants(opt, model):
    mutate_ops = ['NAI', 'NEB', 'GF', 'WS']
    assert(opt.num_model_mutants % len(mutate_ops) == 0)
    for i in range(opt.num_model_mutants + 1):
        mutant = copy.deepcopy(model)
        if i == 0:
            yield i, mutant
        fixed_random = random.Random(opt.seed + i)
        op = mutate_ops[(i - 1) % len(mutate_ops)]
        if op == 'NAI':
            selected_layer = fixed_random.choice([
                m for m in mutant.modules()
                if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh))
            ])
            selected_layer.register_forward_pre_hook(NAI_hook(opt.seed + i))
        else:
            selected_layer = fixed_random.choice([
                m for m in mutant.modules() if hasattr(m, 'weight')
            ])
            if op == 'NEB':
                selected_layer.weight = NEB_hack(selected_layer.weight, opt.seed + i)
            elif op == 'GF':
                selected_layer.weight = GF_hack(selected_layer.weight, opt.seed + i)
            elif op == 'WS':
                selected_layer.weight = WS_hack(selected_layer.weight, opt.seed + i)
        yield i, mutant


def get_input_mutants(opt, input_tensor):
    mutants = []
    mutate_ops = ['PGF', 'PS', 'CPW', 'CPB', 'PCR']
    assert(opt.num_input_mutants % len(mutate_ops) == 0)
    for i in range(opt.num_input_mutants):
        op = mutate_ops[i % len(mutate_ops)]
        if op == 'PGF':
            img_tensor = GF_hack(input_tensor, opt.seed + i, std=0.8, ratio=0.05)
        elif op == 'PS':
            img_tensor = WS_hack(input_tensor, opt.seed + i, ratio=0.05)
        elif op == 'CPW':
            img_tensor = NEB_hack(input_tensor, opt.seed + i, fill=0, ratio=0.05)
        elif op == 'CPB':
            img_tensor = NEB_hack(input_tensor, opt.seed + i, fill=1, ratio=0.05)
        elif op == 'PCR':
            img_tensor = PCR_hack(input_tensor, opt.seed + i, ratio=0.05)
        else:
            raise ValueError('impossible to reach here')
        mutants.append(img_tensor)
    return torch.stack(mutants)


feature_container = []
def feature_hook(module, inputs, outputs):
    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU)):
        mean = torch.mean(outputs, dim=(1, 2, 3))
        feature_container.append(mean)
        var = torch.var(outputs, dim=(1, 2, 3))
        feature_container.append(var)
    if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
        in_mask = (inputs[0] < 0).sum(dim=(1, 2, 3)) / outputs[0].numel()
        out_mask = (outputs < 0).sum(dim=(1, 2, 3)) / outputs[0].numel()
        feature_container.append(out_mask - in_mask)
    elif isinstance(module, nn.Linear):
        mean = torch.mean(inputs[0], dim=1)
        feature_container.append(mean)
        var = torch.var(inputs[0], dim=1)
        feature_container.append(var)
        gini = F.softmax(outputs, dim=1).square().sum(dim=1).mul(-1.).add(1.)
        feature_container.append(gini)


@torch.no_grad()
def extract_features(opt, model, dataloader, device):
    guard_folder(opt, 'extract_features')
    for i, m in tqdm(
            get_model_mutants(opt, model),
            desc='Models', total=opt.num_model_mutants+1):
        filepath = get_output_location(
            opt, ['extract_features', f'features_{i}.pt'])
        if os.path.exists(filepath):
            continue

        for module in m.modules():
            module.register_forward_hook(feature_hook)
        model_mutant = m.to(device)
        model_mutant.eval()

        features, mutation = [], []
        for inputs, targets in tqdm(dataloader, desc='Extract', leave=False):
            assert(inputs.size(0) == 1)
            input_mutants = get_input_mutants(opt, inputs[0]).to(device)
            global feature_container
            feature_container.clear()
            outputs = model_mutant(input_mutants)
            features.append(torch.stack(feature_container, dim=-1).cpu())
            targets = targets.to(device)
            _, predicted = outputs.max(1)
            mutation.append(predicted.eq(targets).cpu())

        features = torch.cat(features)
        mutation = torch.cat(mutation)
        with open(filepath, 'wb') as f:
            torch.save({
                'features': features,
                'mutation': mutation
            }, f)


def main():
    opt = parser.parse_args()
    print(opt)

    device = get_device(opt)
    model = load_model(opt)
    opt.batch_size = 1
    valloader = load_dataloader(opt, split='val')

    extract_features(opt, model, valloader, device)


if __name__ == '__main__':
    main()
