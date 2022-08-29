import time
import copy
import random

import torch
from torch import nn
import torch.utils.data
from tqdm import tqdm

from arguments import parser
from dataset import load_dataloader
from metric import correctness, post_predict, predicates, prediction_error
from model import load_model
from utils import check_file_exists, get_device, guard_folder, rgetattr, rsetattr, save_object


class SilentConv(nn.Module):
    def __init__(self, module, chn):
        super(SilentConv, self).__init__()
        self.module = module
        self.chn = chn

    def forward(self, x):
        out = self.module(x)
        out[:, self.chn] = 0
        return out


class RevertBatchNorm(nn.Module):
    def __init__(self, module, feat_idx):
        super(RevertBatchNorm, self).__init__()
        self.module = module
        self.feat_idx = feat_idx

    def forward(self, x):
        out = self.module(x)
        out[:, self.feat_idx] = x[:, self.feat_idx]
        return out


class RevertReLu(nn.Module):
    def __init__(self, module, seed, device):
        super(RevertReLu, self).__init__()
        self.module = module
        self.seed = seed
        self.device = device

    def forward(self, x):
        out = self.module(x)

        g = torch.Generator(self.device).manual_seed(self.seed)
        mask = torch.bernoulli(
            torch.where(out[0] < 0, 0.1, 0.0), generator=g
        ).bool()
        out = torch.where(mask.expand(out.size()), x[0], out)

        return out


@torch.no_grad()
def evaluate(ctx, model, valloader, device):
    model.eval()

    correct_indicators = []
    label_error, square_error, total = 0, 0, 0
    for inputs, targets in valloader:
        inputs, targets = inputs.to(device), targets.to(device)
        input_preds = post_predict(ctx, model(inputs))

        correct = correctness(ctx, input_preds, targets)
        correct_indicators.append(correct)

        pred_errors = prediction_error(ctx, input_preds, targets)
        if ctx.task == 'clf':
            label_error += pred_errors.sum().item()
        else:
            square_error += pred_errors.square().sum().item()

        total += targets.size(0)

    correct_indicators = torch.cat(correct_indicators)

    if ctx.task == 'clf':
        acc = 100. * (total - label_error) / total
        return acc, correct_indicators
    else:
        mse = square_error / total
        return mse, correct_indicators


emutant_assistant = {}
def get_effimap_model_mutant(ctx, model, device):
    if not hasattr(emutant_assistant, 'init'):
        emutant_assistant['layers'] = [
            (n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU))
        ]
        emutant_assistant['mutant_searched'] = []
        emutant_assistant['init'] = True

    mutant = copy.deepcopy(model).to(device)

    while True:
        lname, layer = random.choice(emutant_assistant['layers'])
        if isinstance(layer, nn.Conv2d):
            chn_idx = random.choice(range(layer.out_channels))
            if f'{lname}_{chn_idx}' in emutant_assistant['mutant_searched']:
                continue
            mlayer = rgetattr(mutant, lname)
            rsetattr(mutant, lname, SilentConv(mlayer, chn_idx))
            emutant_assistant['mutant_searched'].append(f'{lname}_{chn_idx}')
        elif isinstance(layer, nn.BatchNorm2d):
            feat_idx = random.choice(range(layer.num_features))
            if f'{lname}_{feat_idx}' in emutant_assistant['mutant_searched']:
                continue
            mlayer = rgetattr(mutant, lname)
            rsetattr(mutant, lname, RevertBatchNorm(mlayer, feat_idx))
            emutant_assistant['mutant_searched'].append(f'{lname}_{feat_idx}')
        elif isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
            seed = random.randint(0, ctx.num_model_mutants**2)
            if f'{lname}_{seed}' in emutant_assistant['mutant_searched']:
                continue
            mlayer = rgetattr(mutant, lname)
            rsetattr(mutant, lname, RevertReLu(mlayer, seed, device))
            emutant_assistant['mutant_searched'].append(f'{lname}_{seed}')
        else:
            raise ValueError('Invalid layer types')
        break

    return mutant


@torch.no_grad()
def find_killers(ctx, model, mutant, incloader, device):
    model.eval()
    mutant.eval()

    predicate_indicators = []
    for inputs, targets in incloader:
        inputs, targets = inputs.to(device), targets.to(device)
        model_preds = post_predict(ctx, model(inputs))
        mutant_preds = post_predict(ctx, mutant(inputs))

        pdcs = predicates(ctx, model_preds, mutant_preds)
        predicate_indicators.append(pdcs)
    predicate_indicators = torch.cat(predicate_indicators).bool()

    return predicate_indicators


def fuzz_model_mutants(ctx, model, valloader, device):
    perf0, correct_indicators = evaluate(ctx, model, valloader, device)

    incorrect_indice = correct_indicators.nonzero().flatten().tolist()
    incloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(valloader.dataset, incorrect_indice),
        batch_size=ctx.batch_size, shuffle=False, num_workers=8
    )
    correct_indice = correct_indicators.ne(1).nonzero().flatten().tolist()
    corloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(valloader.dataset, correct_indice),
        batch_size=ctx.batch_size, shuffle=False, num_workers=8
    )

    killers = torch.zeros((len(incorrect_indice),), dtype=torch.int)

    mutants = []
    min_energy = 1e9
    start_time = time.time()
    while True:
        elpsed_time = (time.time() - start_time) / 60  # convert to minutes
        if elpsed_time > ctx.fuzz_resource:
            break

        mutant = get_effimap_model_mutant(ctx, model, device)
        perf, _ = evaluate(ctx, mutant, valloader, device)
        if abs(perf - perf0) / perf0 > ctx.perf_tolerance:
            continue

        inc_killers = find_killers(ctx, model, mutant, incloader, device)
        new_cover = torch.logical_and(~killers, inc_killers).sum().gt(0).int()
        cor_killers = find_killers(ctx, model, mutant, corloader, device)
        energy = inc_killers.sum() - cor_killers.sum()

        mutant = mutant.cpu()

        if len(mutants) < ctx.num_model_mutants:
            mutants.append((mutant, new_cover, energy))
            killers = torch.logical_or(killers, inc_killers)
            min_energy = min(energy, min_energy)
            print('appended {}-th mutant'.format(len(mutants)))
            continue

        if new_cover:
            replace_idx = None
            for i, (_, c, _) in enumerate(mutants):
                if not c:
                    replace_idx = i
                    break
            assert(replace_idx is not None)
            mutants[replace_idx] = (mutant, new_cover, energy)
            killers = torch.logical_or(killers, inc_killers)
            min_energy = min([e for _, _, e in mutants])
            print(f'replace {replace_idx}-th mutant with new coverage')
        elif energy > min_energy:
            replace_idx = None
            for i, (_, c, e) in enumerate(mutants):
                if e < energy and c <= new_cover:
                    replace_idx = i
                    break
            if replace_idx is None:
                continue
            mutants[replace_idx] = (mutant, new_cover, energy)
            min_energy = min([e for _, _, e in mutants])
            print(f'replace {replace_idx}-th mutant with higher energy')

    assert(len(mutants) == ctx.num_model_mutants)
    for mutant_idx, (m, _, _) in enumerate(mutants):
        save_object(ctx, m, f'model_mutants/effimap_mutant.{mutant_idx}.pt')


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx, folder='model_mutants')

    device = get_device(ctx)
    valloader = load_dataloader(ctx, split='val')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()

    last_mutant_name = f'model_mutants/fuzz_mutant.{ctx.num_model_mutants-1}.pt'
    if not check_file_exists(ctx, last_mutant_name):
        fuzz_model_mutants(ctx, model, valloader, device)


if __name__ == '__main__':
    main()
