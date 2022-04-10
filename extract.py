import copy
import random

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import load_dataloader
from model import get_device, load_model
from models.vanilla_vae import VanillaVAE
from mutate import *
from arguments import parser
from utils import *


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


@torch.no_grad()
def extract_features(opt, model, dataloader, device):
    guard_folder(opt, 'extract_features')

    model.eval()
    mutate_info = load_object(opt, 'model_mutants_info.json')
    for k, v in mutate_info.items():  # type: ignore
        mutant_idx = int(k[len('mutant'):])  # type: ignore
        lname, chn_idx = v['layer_name'], v['channel_idx']
        module = rgetattr(model, lname)
        module.register_forward_hook(channel_hook(chn_idx, mutant_idx))
    for module in model.modules():
        module.register_forward_hook(feature_hook)

    img_size = next(iter(dataloader))[0].size(-1)
    generator = VanillaVAE(3, img_size, 10).to(device)
    generator.eval()

    result = {}
    filepath = get_output_location(opt, 'extract_features.pt')
    if os.path.exists(filepath):
        result = load_object(opt, filepath)
    for idx, (inputs, targets) in enumerate(
            tqdm(dataloader, desc='Extract', leave=True)):
        if f'sample{idx}' in result.keys(): continue  # type: ignore
        mutation_pool = torch.zeros(
            (1, opt.num_model_mutants), dtype=torch.long, device=device)
        inputs, targets = inputs.to(device), targets.to(device)
        accu_trial = 0
        features, mutation, pred_ret = [], [], []
        with tqdm(total=opt.num_input_mutants, desc='Mutants', leave=False) as pbar:
            while len(features) < opt.num_input_mutants:
                input_mutants = generator.generate(inputs)
                input_mutants = input_mutants.repeat(
                    opt.num_model_mutants + 1, 1, 1, 1)
                feature_container.clear()
                outputs = model(input_mutants)
                _, predicted = outputs.max(1)

                mutated_mask = (predicted[1:] != predicted[0]).long()
                new_pool = torch.cat((mutation_pool, mutated_mask.view(1, -1)))
                if new_pool.std() > mutation_pool.std():
                    mutation_pool = new_pool
                    features.append(torch.stack(feature_container, dim=-1)[0])
                    mutation.append(mutated_mask)
                    pred_ret.append(predicted[0].eq(targets).long())
                    pbar.update(1)
                    accu_trial = 0
                else:
                    accu_trial += 1
                    if accu_trial > opt.num_input_mutants:
                        print(f'Detected no new coverage after {opt.num_input_mutants} trials')
                        mutation_pool.zero_()
                        accu_trial = 0
        result[f'sample{idx}'] = {  # type: ignore
            'features': torch.stack(features).cpu(),
            'mutation': torch.stack(mutation).cpu(),
            'prediction': torch.cat(pred_ret).cpu()
        }
        with open(filepath, 'wb') as f:
            torch.save(result, f)


def main():
    opt = parser.parse_args()
    print(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    print('[info] Enforce to change the batch size to be 1.')
    opt.batch_size = 1
    valloader = load_dataloader(opt, split='val')

    extract_features(opt, model, valloader, device)


if __name__ == '__main__':
    main()
