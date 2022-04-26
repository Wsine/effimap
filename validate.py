import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn import metrics
import numpy as np

from dataset import load_dataloader
from model import get_device, load_model
from mutate import feature_hook, feature_container, channel_hook
from arguments import parser
from utils import *


dispatcher = AttrDispatcher('validator')


@dispatcher.register('furret')
@torch.no_grad()
def estimator_with_mutation_score(opt, model, dataloader, device):
    for module in model.modules():
        module.register_forward_hook(feature_hook)
    model.eval()

    mutation_model = load_object(opt, 'mutation_estimator.pkl')

    df = pd.DataFrame()
    correct, total = 0, 0
    features_pool, equals_pool = [], []
    for inputs, targets in tqdm(dataloader, desc='Validate'):
        inputs, targets = inputs.to(device), targets.to(device)
        feature_container.clear()
        outputs = model(inputs)
        features = torch.stack(feature_container, dim=-1).cpu()
        features_pool.append(features)

        _, predicted = outputs.max(1)
        equals = predicted.eq(targets).cpu()
        equals_pool.append(equals)
        correct += equals.sum().item()
        total += targets.size(0)

    features_pool = torch.cat(features_pool)
    equals_pool = torch.cat(equals_pool)
    mutations = mutation_model.predict(features_pool.numpy())  # type: ignore

    for m, e in zip(mutations, equals_pool):
        df = df.append({
            'furret': m.sum(),
            'actual': e.item()
        }, ignore_index=True)

    # statistic result
    print('test acc: {:.2f}%'.format(100. * correct / total))
    df = df.astype({'furret': float, 'actual': int})
    # for t in range(5, opt.num_model_mutants, 5)[::-1]:
    for t in range(1, 20 + 1)[::-1]:
        sub_df = df.query("furret < {}".format(t))
        print('test acc of {}: {:.2f}%'.format(
              t, 100. * sub_df['actual'].sum() / len(sub_df)))  # type: ignore


@dispatcher.register('dissector')
@torch.no_grad()
def dissector_method(opt, model, dataloader, device):
    model.eval()

    hook_vec = {}
    def _hook_on_layer(lname):
        def __hook(module, inputs, outputs):
            hook_vec[lname] = outputs.detach().flatten(start_dim=1)
        return __hook

    snapshotors = load_object(opt, 'snapshotors.pt')
    for k in snapshotors.keys():  # type: ignore
        module = rgetattr(model, k)
        module.register_forward_hook(_hook_on_layer(k))
    batch_inputs, _ = next(iter(dataloader))
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

    df = pd.DataFrame()
    for inputs, targets in tqdm(dataloader, desc='Validate'):
        inputs, targets = inputs.to(device), targets.to(device)
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


        equals = predicted.eq(targets)
        for p, e in zip(pvscore, equals):
            df = df.append({
                'dissector': p.item(),
                'actual': e.item()
            }, ignore_index=True)

    df = df.astype({'dissector': float, 'actual': int})
    for t in range(5, opt.num_model_mutants, 5):
        sub_df = df.query("dissector > {}".format(t * 0.01))
        print('test acc of {}: {:.2f}%'.format(
              t, 100. * sub_df['actual'].sum() / len(sub_df)))  # type: ignore


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    testloader = load_dataloader(opt, split='test')

    dispatcher(opt, model, testloader, device)


if __name__ == '__main__':
    main()
