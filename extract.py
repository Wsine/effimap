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
                m for m in mutant.modules() if hasattr(m, 'weight') and torch.is_tensor(m.weight)
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


@dispatcher.register('prima')
@torch.no_grad()
def prima_extract_features(opt, model, device):
    model.eval()
    valloader = load_dataloader(opt, split=opt.prima_split)

    input_feat_path = get_output_location(
            opt, f'prima_input_features_{opt.prima_split}.pt')
    if not os.path.exists(input_feat_path):
        input_features = []
        for inputs, _ in tqdm(valloader, desc='Input Mutants'):
            if opt.task == 'regress':
                pred0 = model(inputs.to(device)).flatten()
            else:
                output0 = model(inputs.to(device))
                prob0 = F.softmax(output0, dim=1)
                _, pred0 = output0.max(1)
            for i, input in enumerate(inputs):
                input_mutants = get_input_mutants(opt, input).to(device)
                if opt.task == 'regress':
                    pred = model(input_mutants).flatten()
                    diff = (pred - pred0[i]).abs()
                    f2a = diff.mean()
                    f2b = [
                        torch.logical_and(diff > s, diff < s + 0.1).sum()
                        for s in torch.linspace(0, 1, steps=10)
                    ]
                    feat = torch.Tensor([f2a, *f2b])
                else:
                    output = model(input_mutants)
                    prob = F.softmax(output, dim=1)
                    _, pred = output.max(1)
                    f1a = pred.ne(pred0[i]).sum()
                    f1b = pred.unique().size(0) - 1
                    _, cnt = pred.unique(return_counts=True)
                    f1c = cnt[cnt.topk(2).indices[-1]] if cnt.size(0) > 1 else cnt[0]
                    dist = F.cosine_similarity(
                        prob, prob0[i].repeat(prob.size(0), 1))  # type: ignore
                    f2a = dist.mean()
                    f2b = [
                        torch.logical_and(dist > s, dist < s + 0.1).sum()
                        for s in torch.linspace(0, 1, steps=10)
                    ]
                    f2c = (prob[:, pred0[i]] - prob0[i, pred0[i]]).mean()  # type: ignore
                    feat = torch.Tensor([f1a, f1b, f1c, f2a, *f2b, f2c])
                input_features.append(feat.cpu())
        input_features = torch.stack(input_features)
        torch.save({'feats': input_features}, input_feat_path)

    feat_target_path = get_output_location(
            opt, f'prima_feature_target_{opt.prima_split}.pt')
    model_feat_path = get_output_location(
            opt, f'prima_model_features_{opt.prima_split}.pt')
    if not os.path.exists(model_feat_path):
        prob0, pred0, equals = [], [], []
        for inputs, targets in tqdm(valloader, desc='Original Model'):
            inputs, targets = inputs.to(device), targets.to(device)
            if opt.task == 'regress':
                prob0.append(torch.zeros((inputs.size(0), opt.num_classes)))  # useless
                pred = model(inputs).flatten()
                pred0.append(pred)
                delta = pred.view(-1) - targets.view(-1)
                equals.append(delta.abs())
            else:
                output = model(inputs)
                prob = F.softmax(output, dim=1)
                prob0.append(prob)
                _, pred = output.max(1)
                pred0.append(pred)
                equals.append(pred.eq(targets))
        equals = torch.cat(equals).cpu()
        torch.save({'equals': equals}, feat_target_path)

        model_probs, model_preds = [], []
        for mutant in tqdm(
                get_model_mutants(opt, model.cpu()),
                desc='Model Mutants', total=opt.num_model_mutants):
            mutant.eval()
            mutant = mutant.to(device)
            probs, preds = [], []
            for inputs, _ in tqdm(valloader, desc='Inference', leave=False):
                if opt.task == 'regress':
                    prob = torch.zeros((inputs.size(0), opt.num_classes))  # useless
                    pred = mutant(inputs.to(device)).flatten()
                else:
                    output = mutant(inputs.to(device))
                    prob = F.softmax(output, dim=1)
                    _, pred = output.max(1)
                probs.append(prob)
                preds.append(pred)
            model_probs.append(torch.cat(probs))
            model_preds.append(torch.cat(preds))
        model_probs = torch.stack(model_probs, dim=1)
        model_preds = torch.stack(model_preds, dim=-1)

        prob0 = torch.cat(prob0)
        pred0 = torch.cat(pred0)
        model_features = []
        for prob, pred, po0, pe0 in zip(model_probs, model_preds, prob0, pred0):
            if opt.task == 'regress':
                diff = (pred - pe0).abs()
                f2a = diff.mean()
                f2b = [
                    torch.logical_and(diff > s, diff < s + 0.1).sum()
                    for s in torch.linspace(0, 1, steps=10)
                ]
                feat = torch.Tensor([f2a, *f2b])
            else:
                f1a = pred.ne(pe0).sum()
                f1b = pred.unique().size(0) - 1
                _, cnt = pred.unique(return_counts=True)
                f1c = cnt[cnt.topk(2).indices[-1]] if cnt.size(0) > 1 else cnt[0]
                dist = F.cosine_similarity(prob, po0.repeat(prob.size(0), 1))
                f2a = dist.mean()
                f2b = [
                    torch.logical_and(dist > s, dist < s + 0.1).sum()
                    for s in torch.linspace(0, 1, steps=10)
                ]
                f2c = (prob[:, pe0] - po0[pe0]).mean()
                feat = torch.Tensor([f1a, f1b, f1c, f2a, *f2b, f2c])
            model_features.append(feat.cpu())
        model_features = torch.stack(model_features)
        torch.save({'feats': model_preds}, model_feat_path)


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
