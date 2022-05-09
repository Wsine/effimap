import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_dataloader
from model import get_device, load_model
from mutate import channel_hook
from extract import get_model_mutants
from arguments import parser
from utils import *


@torch.no_grad()
def export_effimap_mutation_analysis(opt, model, device):
    print('[info] Enforce to change the batch size to be 1.')
    opt.batch_size = 1
    testloader = load_dataloader(opt, split='test')

    model.eval()
    mutate_info = load_object(opt, 'model_mutants_info.json')
    for k, v in mutate_info.items():  # type: ignore
        mutant_idx = int(k[len('mutant'):])  # type: ignore
        lname, chn_idx = v['layer_name'], v['channel_idx']
        module = rgetattr(model, lname)
        module.register_forward_hook(channel_hook(chn_idx, mutant_idx))

    correct, total = 0, 0
    mutated_pool, equals_pool = [], []
    for inputs, targets in tqdm(testloader, desc='Export'):
        inputs, targets = inputs.to(device), targets.to(device)
        input_repeats = inputs.repeat(opt.num_model_mutants + 1, 1, 1, 1)
        outputs = model(input_repeats)

        if opt.task == 'regress':
            predicted = outputs.flatten()
            mse = torch.tensor(0.85 if opt.dataset == 'mnist' else 1.40)
            mutated_mask = (predicted[1:] - predicted[0]).abs() > predicted[0].mul(0.05)
            mutated_pool.append(mutated_mask.long().cpu())
            equal = (predicted[0] - targets[0]).abs() < mse.mul(1.05)
            equals_pool.append(equal.long().cpu())
            correct += (predicted[0] - targets[0]).abs().item()
        else:
            _, predicted = outputs.max(1)
            mutated_mask = (predicted[1:] != predicted[0])
            mutated_pool.append(mutated_mask.long().cpu())
            equal = predicted[0].eq(targets[0])
            equals_pool.append(equal.long().cpu())
            correct += equal.item()
        total += 1

    acc = 100. * correct / total
    print('test acc = {:.2f}'.format(acc))

    result = {
        'mutated_pool': torch.stack(mutated_pool, dim=-1),
        'equals_pool': torch.Tensor(equals_pool)
    }
    print('[debug] mutated_pool size', result['mutated_pool'].size())
    print('[debug] equals_pool size', result['equals_pool'].size())
    print(result['equals_pool'].sum() / result['equals_pool'].size(0))
    debug = result['mutated_pool'].sum(dim=1).bool().long()
    print(debug.sum() / debug.size(0))

    torch.save(result, get_output_location(opt, f'{opt.dataset}_EffiMAP_mutation_analysis.pt'))


@torch.no_grad()
def export_prima_mutation_analysis(opt, model, device):
    model.eval()
    testloader = load_dataloader(opt, split='test')

    mse = torch.tensor(0.85 if opt.dataset == 'mnist' else 1.40)
    preds_pool, equals_pool = [], []
    for inputs, targets in tqdm(testloader, desc='Original Model'):
        inputs, targets = inputs.to(device), targets.to(device)
        if opt.task == 'regress':
            outputs = model(inputs)
            preds = outputs.flatten()
            preds_pool.append(preds)
            equals = (preds - targets).abs() < mse.mul(1.05)
            equals_pool.append(equals.long().cpu())
        else:
            outputs = model(inputs)
            _, preds = outputs.max(1)
            preds_pool.append(preds)
            equals = preds.eq(targets)
            equals_pool.append(equals.long().cpu())
    equals_pool = torch.cat(equals_pool)
    preds_pool = torch.cat(preds_pool)

    mutated_pool = []
    for mutant in tqdm(
            get_model_mutants(opt, model.cpu()),
            desc='Model Mutants', total=opt.num_model_mutants):
        mutant.eval()
        mutant = mutant.to(device)
        preds_seq = []
        for inputs, _ in tqdm(testloader, desc='Inference', leave=False):
            if opt.task == 'regress':
                outputs = mutant(inputs.to(device))
                preds = outputs.flatten()
                preds_seq.append(preds)
            else:
                outputs = mutant(inputs.to(device))
                _, preds = outputs.max(1)
                preds_seq.append(preds)
        preds_seq = torch.cat(preds_seq)
        if opt.task == 'regress':
            mutated_mask = (preds_seq - preds_pool).abs() > preds_pool.mul(0.05)
        else:
            mutated_mask = preds_seq.eq(preds_pool)
        mutated_pool.append(mutated_mask.long().cpu())

    result = {
        'mutated_pool': torch.stack(mutated_pool),
        'equals_pool': equals_pool
    }
    print('[debug] mutated_pool size', result['mutated_pool'].size())
    print('[debug] equals_pool size', result['equals_pool'].size())

    torch.save(result, get_output_location(opt, f'{opt.dataset}_PRIMA_mutation_analysis.pt'))


@torch.no_grad()
def export_dissector_ranking_list(opt, model, device):
    model.eval()
    testloader = load_dataloader(opt, split='test')

    hook_vec = {}
    def _hook_on_layer(lname):
        def __hook(module, inputs, outputs):
            hook_vec[lname] = outputs.detach().flatten(start_dim=1)
        return __hook

    snapshotors = load_object(opt, 'snapshotors.pt')
    for k in snapshotors.keys():  # type: ignore
        module = rgetattr(model, k)
        module.register_forward_hook(_hook_on_layer(k))
    batch_inputs, _ = next(iter(testloader))
    model(batch_inputs.to(device))
    hook_snapshotor = {}
    for k in snapshotors.keys():  # type: ignore
        snapshotor = torch.nn.Linear(hook_vec[k].size(1), opt.num_classes).to(device)
        snapshotor.load_state_dict(snapshotors[k]['net'])  # type: ignore
        snapshotor.eval()
        hook_snapshotor[k] = snapshotor

    if opt.dataset == 'cifar100' and opt.model == 'resnet32':
        snap_order = ['relu', 'layer1', 'layer2', 'layer3']
    elif opt.dataset == 'tinyimagenet' and opt.model == 'resnet18':
        snap_order = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
    else:
        raise ValueError('Not supported combinations now')

    pvscores, equals_pool = [], []
    for inputs, targets in tqdm(testloader, desc='Export'):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        snapshot = torch.stack([
            F.softmax(hook_snapshotor[k](hook_vec[k]), dim=1)
            for k in snap_order
        ], dim=-1)
        highest = snapshot.max(1).values
        second_highest = snapshot.topk(2, dim=1).values[:, 1, :]
        predicted_high = torch.stack([snapshot[i, p, :] for i, p in enumerate(predicted)])
        svscore1 = highest / (highest + second_highest)
        svscore2 = 1 - highest / (predicted_high + highest)
        mask = snapshot.max(1).indices == \
               predicted.repeat(snapshot.size(-1), 1).transpose(0, 1)
        svscore = torch.where(mask, svscore1, svscore2)
        weights = torch.log(torch.arange(1, svscore.size(1) + 1, device=device))
        pvscore = (svscore * weights).sum(dim=1) / weights.sum()
        pvscores.append(pvscore)
        equals = predicted.eq(targets.to(device))
        equals_pool.append(equals)
    pvscores = torch.cat(pvscores)
    equals_pool = torch.cat(equals_pool)

    _, indices = torch.sort(pvscores, descending=False)
    indices = indices.cpu()
    print('[debug] indices size: ', indices.size())

    print(equals_pool[indices].tolist())

    torch.save(indices, get_output_location(opt, f'{opt.dataset}_DISSECTOR_rank_indices.pt'))


@torch.no_grad()
def export_ideal_ranking_list(opt, model, device):
    model.eval()
    testloader = load_dataloader(opt, split='test')

    delta_pool = []
    for inputs, targets in tqdm(testloader, desc='Export'):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predicted = outputs.flatten()
        delta = predicted.view(-1) - targets.view(-1)
        delta_pool.append(delta.abs().cpu())
    delta_pool = torch.cat(delta_pool)

    _, indices = torch.sort(delta_pool, descending=True)
    print('[debug] indices size: ', indices.size())
    print(delta_pool[indices].tolist())

    torch.save(indices, get_output_location(opt, f'{opt.dataset}_IDEAL_rank_indices.pt'))


def main():
    opt = parser.parse_args()
    print(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)

    # export_effimap_mutation_analysis(opt, model, device)
    # export_prima_mutation_analysis(opt, model, device)
    # export_dissector_ranking_list(opt, model, device)
    export_ideal_ranking_list(opt, model, device)
    print('Done')


if __name__ == '__main__':
    main()
