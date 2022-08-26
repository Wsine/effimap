import copy
import random

import torch
from torch import nn
from tqdm import tqdm

from arguments import parser
from model import load_model
from dataset import load_dataloader
from utils import check_file_exists, guard_folder, rsetattr, save_object


def get_tensor_mask(tensor, seed=None, ratio=0.1):
    g = torch.Generator().manual_seed(seed) if seed else None
    num_neurons = tensor.numel()
    num_mask = int(num_neurons * ratio)
    shuffle = torch.randperm(num_neurons, generator=g)
    mask = torch.cat(
        (torch.ones((num_mask,), dtype=torch.long),
         torch.zeros((num_neurons - num_mask,), dtype=torch.long))
    )[shuffle].reshape_as(tensor).bool()
    return mask


class InverseActivate(nn.Module):
    def __init__(self, module, seed):
        super(InverseActivate, self).__init__()
        self.module = module
        self.seed = seed

    def forward(self, x):
        mask = get_tensor_mask(x[0], self.seed)
        try:
            device = x.get_device()
            mask = mask.cuda(device)
        except RuntimeError:
            pass
        for i, img in enumerate(x):
            x[i] = torch.where(mask, img.mul(-1.), img)
        return x


def NEB_hack(tensor, fill=0, **kwargs):
    mask = get_tensor_mask(tensor, **kwargs)
    tensor.masked_fill_(mask, fill)
    return tensor


def GF_hack(tensor, std=0.1, **kwargs):
    mask = get_tensor_mask(tensor, **kwargs)
    gauss = torch.normal(0, std, tensor.size())
    if isinstance(tensor, nn.parameter.Parameter):
        tensor.copy_(torch.where(mask, tensor + gauss, tensor))
    else:
        tensor = torch.where(mask, tensor + gauss, tensor)
    return tensor


def WS_hack(tensor, ratio=0.1):
    mask = get_tensor_mask(tensor, ratio=ratio)
    mask_indices = mask.nonzero(as_tuple=True)
    shuffle = torch.randperm(mask_indices[0].size(0))
    tensor[mask_indices] = tensor[mask_indices][shuffle]
    return tensor


def PCR_hack(tensor, **kwargs):
    mask = get_tensor_mask(tensor, **kwargs)
    tensor = torch.where(mask, 1 - tensor, tensor)
    return tensor


@torch.no_grad()
def generate_random_model_mutants(ctx, model):
    mutate_ops = ['NAI', 'NEB', 'GF', 'WS']
    assert(ctx.num_model_mutants % len(mutate_ops) == 0)

    mutant_idx = 0
    while (mutant_idx < ctx.num_model_mutants):
        mutant = copy.deepcopy(model)

        op = random.choice(mutate_ops)
        if op == 'NAI':
            modules = [
                (n, m) for n, m in mutant.named_modules()
                if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh))
            ]
            if len(modules) == 0:
                print('This model contains no activation layers')
                continue
            selected_layer = random.choice(modules)
            rsetattr(mutant, selected_layer[0], InverseActivate(selected_layer[1], mutant_idx))
        else:
            selected_layer = random.choice([
                m for m in mutant.modules() if hasattr(m, 'weight') and torch.is_tensor(m.weight)
            ])
            if op == 'NEB':
                selected_layer.weight = NEB_hack(selected_layer.weight)
            elif op == 'GF':
                selected_layer.weight = GF_hack(selected_layer.weight)
            elif op == 'WS':
                selected_layer.weight = WS_hack(selected_layer.weight)

        save_object(ctx, mutant, f'model_mutants/random_mutant.{mutant_idx}.pt')
        mutant_idx += 1


def generate_random_sample_mutants(ctx, batch_inputs):
    mutate_ops = ['PGF', 'PS', 'CPW', 'CPB', 'PCR']
    assert(ctx.num_sample_mutants % len(mutate_ops) == 0)

    for sample in tqdm(batch_inputs, desc='Samples', leave=False):
        sample_mutants = [sample]  # 0 stays the original sample
        for _ in range(ctx.num_sample_mutants):
            op = random.choice(mutate_ops)
            if op == 'PGF':
                mutant = GF_hack(sample, std=0.8, ratio=0.05)
            elif op == 'PS':
                mutant = WS_hack(sample, ratio=0.05)
            elif op == 'CPW':
                mutant = NEB_hack(sample, fill=0, ratio=0.05)
            elif op == 'CPB':
                mutant = NEB_hack(sample, fill=1, ratio=0.05)
            elif op == 'PCR':
                mutant = PCR_hack(sample, ratio=0.05)
            else:
                raise ValueError('Invalid operator')
            sample_mutants.append(mutant)
        sample_with_mutants = torch.stack(sample_mutants)

        yield sample_with_mutants


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx, folder=['model_mutants', 'sample_mutants'])

    last_mutant_name = f'model_mutants/random_mutant.{ctx.num_model_mutants-1}.pt'
    if not check_file_exists(ctx, last_mutant_name):
        model = load_model(ctx, pretrained=True)
        model.eval()
        generate_random_model_mutants(ctx, model)

    last_mutant_name = f'sample_mutants/random_mutant.{ctx.num_sample_mutants-1}.pt'
    if not check_file_exists(ctx, last_mutant_name):
        valloader = load_dataloader(ctx, split='val')
        generate_random_sample_mutants(ctx, valloader)

if __name__ == '__main__':
    main()
