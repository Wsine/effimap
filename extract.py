import copy
import random

import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

from dataset import load_dataloader
from model import get_device, load_model
from models.vanilla_vae import VanillaVAE
from mutate import *
from arguments import parser
from utils import *


dispatcher = AttrDispatcher('strategy')


def get_model_mutants(opt, model):
    mutate_ops = ['NAI', 'NEB', 'GF', 'WS']
    assert(opt.num_model_mutants % len(mutate_ops) == 0)
    for i in range(opt.num_model_mutants):
        mutant = copy.deepcopy(model)
        fixed_random = random.Random(opt.seed + i)
        op = mutate_ops[i % len(mutate_ops)]
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
        yield mutant


def get_input_mutants(opt, input_tensor):
    mutants = []
    mutate_ops = ['PGF', 'PS', 'CPW', 'CPB', 'PCR']
    assert(opt.num_input_mutants % len(mutate_ops) == 0)
    for i in range(opt.num_input_mutants):
        #  op = mutate_ops[i % len(mutate_ops)]
        op = random.choice(mutate_ops)
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


@dispatcher.register('random')
@torch.no_grad()
def prima_extract_features(opt, model, device):
    model.eval()
    valloader = load_dataloader(opt, split=opt.prima_split)

    input_feat_path = get_output_location(
            opt, f'prima_input_features_{opt.prima_split}.pt')
    if not os.path.exists(input_feat_path):
        input_features = []
        for inputs, _ in tqdm(valloader, desc='Input Mutants'):
            _, pred0 = model(inputs.to(device)).max(1)
            for i, input in enumerate(inputs):
                input_mutants = get_input_mutants(opt, input).to(device)
                _, pred = model(input_mutants).max(1)
                mutated = pred.ne(pred0[i]).int()
                input_features.append(mutated.cpu())
        input_features = torch.stack(input_features)
        torch.save({'feats': input_features}, input_feat_path)

    feat_target_path = get_output_location(
            opt, f'prima_feature_target_{opt.prima_split}.pt')
    model_feat_path = get_output_location(
            opt, f'prima_model_features_{opt.prima_split}.pt')
    if not os.path.exists(model_feat_path):
        pred0, equals = [], []
        for inputs, targets in tqdm(valloader, desc='Original Model'):
            inputs, targets = inputs.to(device), targets.to(device)
            _, pred = model(inputs).max(1)
            pred0.append(pred)
            equals.append(pred.eq(targets))
        equals = torch.cat(equals).cpu()
        torch.save({'equals': equals}, feat_target_path)

        model_features = []
        for mutant in tqdm(
                get_model_mutants(opt, model.cpu()),
                desc='Model Mutants', total=opt.num_model_mutants):
            mutant.eval()
            mutant = mutant.to(device)
            mutated = []
            for batch_idx, (inputs, _) in enumerate(
                    tqdm(valloader, desc='Inference', leave=False)):
                _, pred = mutant(inputs.to(device)).max(1)
                mutated.append(pred.ne(pred0[batch_idx]).cpu())
            model_features.append(torch.cat(mutated))
        model_features = torch.stack(model_features, dim=-1)
        torch.save({'feats': model_features}, model_feat_path)


@dispatcher.register('furret')
@torch.no_grad()
def furret_extract_features(opt, model, device):
    print('[info] Enforce to change the batch size to be 1.')
    opt.batch_size = 1
    valloader = load_dataloader(opt, split='val')

    model.eval()
    mutate_info = load_object(opt, 'model_mutants_info.json')
    for k, v in mutate_info.items():  # type: ignore
        mutant_idx = int(k[len('mutant'):])  # type: ignore
        lname, chn_idx = v['layer_name'], v['channel_idx']
        module = rgetattr(model, lname)
        module.register_forward_hook(channel_hook(chn_idx, mutant_idx))
    for module in model.modules():
        module.register_forward_hook(feature_hook)

    sample_img = next(iter(valloader))[0][0]
    img_channels, img_size = sample_img.size(0), sample_img.size(-1)
    if img_size < 32:
        pad = T.Pad((32 - img_size) // 2)
        crop = T.CenterCrop(img_size)
        img_size = 32
    else:
        pad = crop = None
    generator = VanillaVAE(img_channels, img_size, 10)
    state = load_object(opt, 'encoder_model.pt')
    generator.load_state_dict(state['net'])  # type: ignore
    generator = generator.to(device)
    generator.eval()

    result = {}
    filepath = get_output_location(opt, 'extract_features.pt')
    if os.path.exists(filepath):
        result = load_object(opt, 'extract_features.pt')
    for idx, (inputs, targets) in enumerate(
            tqdm(valloader, desc='Extract', leave=True)):
        if f'sample{idx}' in result.keys(): continue  # type: ignore
        inputs, targets = inputs.to(device), targets.to(device)
        features, mutation, pred_ret = [], [], []
        with tqdm(total=opt.num_input_mutants, desc='Mutants', leave=False) as pbar:
            while len(features) < opt.num_input_mutants:
                if pad is not None and crop is not None:
                    input_mutants = crop(generator.generate(pad(inputs)))
                else:
                    input_mutants = generator.generate(inputs)
                input_mutants = input_mutants.repeat(
                    opt.num_model_mutants + 1, 1, 1, 1)

                feature_container.clear()
                outputs = model(input_mutants)

                if opt.task == 'regress':
                    predicted = outputs.flatten()
                    mutated = predicted[1:] - predicted[0]
                    mutation.append(mutated.abs())
                    gt_ret = predicted[0].view(-1) - targets.view(-1)
                    pred_ret.append(gt_ret.abs())
                else:
                    _, predicted = outputs.max(1)
                    mutated_mask = (predicted[1:] != predicted[0])
                    mutation.append(mutated_mask.long())
                    pred_ret.append(predicted[0].eq(targets).long())

                features.append(torch.stack(feature_container, dim=-1)[0])
                pbar.update(1)
        result[f'sample{idx}'] = {  # type: ignore
            'features': torch.stack(features).cpu(),
            'mutation': torch.stack(mutation).cpu(),
            'prediction': torch.cat(pred_ret).cpu()
        }
        with open(filepath, 'wb') as f:
            torch.save(result, f)


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)
    guard_folder(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)

    dispatcher(opt, model, device)


if __name__ == '__main__':
    main()
