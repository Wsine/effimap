import time
import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn import metrics
import numpy as np

from dataset import load_dataloader
from model import get_device, load_model
from models.vanilla_vae import VanillaVAE
from mutate import feature_hook, feature_container, channel_hook
from arguments import parser
from utils import *


dispatcher = AttrDispatcher('prioritor')


def auc_for_regression(y):
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()
    area_y = np.trapz(np.cumsum(y))
    z = np.cumsum(np.sort(y)[::-1])
    max_area_y = np.trapz(z)
    return area_y / max_area_y


@dispatcher.register('gini')
@torch.no_grad()
def gini_method(_, model, dataloader, device):
    model.eval()

    df = pd.DataFrame()
    correct, total = 0, 0
    for inputs, targets in tqdm(dataloader, desc='Validate'):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        _, predicted = outputs.max(1)
        equals = predicted.eq(targets)
        correct += equals.sum().item()
        total += targets.size(0)

        gini = F.softmax(outputs, dim=1).square().sum(dim=1).mul(-1.).add(1.)

        for g, e in zip(gini, equals):
            df = df.append({
                'gini': g.item(),
                'actual': (~e).item()
            }, ignore_index=True)

    # statistic result
    print('test acc: {:.2f}%'.format(100. * correct / total))

    df = df.astype({'gini': float, 'actual': int})
    fpr, tpr, _ = metrics.roc_curve(df['actual'], df['gini'])
    auc = metrics.auc(fpr, tpr)
    print('auc for gini: {:.2f}%'.format(100. * auc))


@dispatcher.register('furret')
@torch.no_grad()
def estimator_with_mutation_score(opt, model, dataloader, device):
    for module in model.modules():
        module.register_forward_hook(feature_hook)
    model.eval()

    mutation_model = load_object(opt, 'mutation_estimator.pkl')

    start_time = time.time()

    df = pd.DataFrame()
    correct, total = 0, 0
    features_pool, equals_pool = [], []
    for inputs, targets in tqdm(dataloader, desc='Validate'):
        inputs, targets = inputs.to(device), targets.to(device)
        feature_container.clear()
        outputs = model(inputs)
        features = torch.stack(feature_container, dim=-1).cpu()
        features_pool.append(features)

        if opt.task == 'regress':
            predicted = outputs.flatten()
            delta = predicted.view(-1) - targets.view(-1)
            equals_pool.append(delta.abs().cpu())
            correct += delta.abs().sum().item()
        else:
            _, predicted = outputs.max(1)
            equals = predicted.eq(targets).cpu()
            equals_pool.append(equals)
            correct += equals.sum().item()
        total += targets.size(0)

    features_pool = torch.cat(features_pool)
    equals_pool = torch.cat(equals_pool)
    second_time = time.time()
    mutations = mutation_model.predict(features_pool.numpy())  # type: ignore

    for m, e in zip(mutations, equals_pool):
        df = df.append({
            'furret': m.sum(),
            'actual': e.item() if opt.task == 'regress' else (~e).item()
        }, ignore_index=True)

    # statistic result
    if opt.task == 'regress':
        print('test mse: {:.8f}'.format(correct / total))
        df = df.astype({'furret': float, 'actual': float})
        df.sort_values(by=['furret'], ascending=True, inplace=True)
        for r in [0.1, 0.2, 0.3, 0.5]:
            sub_seq = int(df.size * r)
            auc = auc_for_regression(df.head(sub_seq)['actual'])
            print('rauc for furret {:.2f}%: {:.2f}%'.format(r * 100, 100. * auc))
        auc = auc_for_regression(df['actual'])
        print('auc for mutation score: {:.2f}%'.format(100. * auc))
    else:
        print('test acc: {:.2f}%'.format(100. * correct / total))
        df = df.astype({'furret': float, 'actual': int})
        df.sort_values(by=['furret'], ascending=True, inplace=True)
        for r in [0.1, 0.2, 0.3, 0.5]:
            sub_seq = int(df.size * r)
            fpr, tpr, _ = metrics.roc_curve(
                df.head(sub_seq)['actual'], df.head(sub_seq)['furret'])
            auc = metrics.auc(fpr, tpr)
            print('rauc for furret {:.2f}%: {:.2f}%'.format(r * 100, 100. * auc))
        fpr, tpr, _ = metrics.roc_curve(df['actual'], df['furret'])
        auc = metrics.auc(fpr, tpr)
        print('rauc for furret all: {:.2f}%'.format(100. * auc))
    third_time = time.time()
    print('time for feature extraction = ', second_time - start_time)
    print('time for learing to rank = ', third_time - second_time)


@dispatcher.register('furret2')
@torch.no_grad()
def estimator_with_multi_mutation_score(opt, model, dataloader, device):
    model.eval()
    mutated_model = copy.deepcopy(model)
    mutate_info = load_object(opt, 'model_mutants_info.json')
    for k, v in mutate_info.items():  # type: ignore
        mutant_idx = int(k[len('mutant'):])  # type: ignore
        lname, chn_idx = v['layer_name'], v['channel_idx']
        module = rgetattr(mutated_model, lname)
        module.register_forward_hook(channel_hook(chn_idx, mutant_idx - 1))
    for module in mutated_model.modules():
        module.register_forward_hook(feature_hook)
    mutated_model.eval()

    img_size = next(iter(dataloader))[0].size(-1)
    generator = VanillaVAE(3, img_size, 10)
    state = load_object(opt, 'encoder_model.pt')
    generator.load_state_dict(state['net'])  # type: ignore
    generator = generator.to(device)
    generator.eval()

    mutation_model = load_object(opt, 'mutation_estimator.pkl')

    df = pd.DataFrame()
    for inputs, targets in tqdm(dataloader, desc='Validate'):
        inputs, targets = inputs.to(device), targets.to(device)
        _, predicted = model(inputs).max(1)
        equals = predicted.eq(targets)

        input_features_pool, model_features_pool = [], []
        for idx, input in enumerate(tqdm(inputs, desc='Samples', leave=False)):
            input_repeat = input.view(1, *input.size()) \
                    .repeat(opt.num_input_mutants, 1, 1, 1)
            _, pred = model(input_repeat).max(1)
            input_features_pool.append(pred.ne(predicted[idx]).cpu())

            input_repeat = input.view(1, *input.size()) \
                    .repeat(opt.num_model_mutants, 1, 1, 1)
            input_mutants = generator.generate(input_repeat)
            feature_container.clear()
            mutated_model(input_mutants)
            features = torch.stack(feature_container, dim=-1).cpu()
            model_features_pool.append(features)
        model_features_pool = torch.cat(model_features_pool)

        mutation = mutation_model.predict(model_features_pool.numpy())  # type: ignore
        for i in range(equals.size(0)):
            f1 = input_features_pool[i].sum().item()
            f2 = mutation[i*opt.num_model_mutants:(i+1)*opt.num_model_mutants].sum()
            df = df.append({
                'mutation_score': f1 + f2,
                'actual': (~equals[i]).item()
            }, ignore_index=True)

    df = df.astype({'mutation_score': float, 'actual': int})
    fpr, tpr, _ = metrics.roc_curve(df['actual'], df['mutation_score'])
    auc = metrics.auc(fpr, tpr)
    print('auc for mutation score: {:.2f}%'.format(100. * auc))


@dispatcher.register('prima')
def prima_method(opt, *_):
    input_features = load_object(opt, 'prima_input_features_test.pt')
    model_features = load_object(opt, 'prima_model_features_test.pt')
    feature_target = load_object(opt, 'prima_feature_target_test.pt')
    X = torch.cat(
        (input_features['feats'], model_features['feats']),  # type: ignore
        dim=1).numpy()
    Y = feature_target['equals'].numpy()  # type: ignore
    print('[info] data loaded.')

    ranking_model = load_object(opt, 'prima_ranking_model.pkl')
    start_time = time.time()
    rank = ranking_model.predict(X)  # type: ignore
    second_time = time.time()

    if opt.task == 'regress':
        sort_inds = rank.argsort()
        Y = Y[sort_inds]
        for r in [0.1, 0.2, 0.3, 0.5]:
            sub_Y = Y[:int(len(Y) * r)]
            auc = auc_for_regression(sub_Y)
            print('rauc for prima {:.2f}%: {:.2f}%'.format(r * 100, 100. * auc))
        auc = auc_for_regression(Y)
        print('rauc for prima all: {:.2f}%'.format(100. * auc))
    else:
        sort_inds = rank.argsort()
        Y = Y[sort_inds]
        rank = rank[sort_inds]
        for r in [0.1, 0.2, 0.3, 0.5]:
            sub_seq = int(len(Y) * r)
            fpr, tpr, _ = metrics.roc_curve(Y[:sub_seq], rank[:sub_seq])
            auc = metrics.auc(fpr, tpr)
            print('rauc for prima {:.2f}%: {:.2f}%'.format(r * 100, 100. * auc))
        fpr, tpr, _ = metrics.roc_curve(Y, rank)
        auc = metrics.auc(fpr, tpr)
        print('rauc for prima all: {:.2f}%'.format(100. * auc))
    third_time = time.time()
    print('time for feature extraction = ', second_time - start_time)
    print('time for learing to rank = ', third_time - second_time)


@dispatcher.register('dissector')
@torch.no_grad()
def dissector_method(opt, model, dataloader, device):
    model.eval()

    start_time = time.time()

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
    df.sort_values(by=['dissector'], ascending=True, inplace=True)
    for r in [0.1, 0.2, 0.3, 0.5]:
        sub_seq = int(df.size * r)
        fpr, tpr, _ = metrics.roc_curve(
            df.head(sub_seq)['actual'], df.head(sub_seq)['dissector'])
        auc = metrics.auc(fpr, tpr)
        print('rauc for dissector {:.2f}%: {:.2f}%'.format(r * 100, 100. * auc))
    fpr, tpr, _ = metrics.roc_curve(df['actual'], df['dissector'])
    auc = metrics.auc(fpr, tpr)
    print('rauc for dissector all: {:.2f}%'.format(100. * auc))

    second_time = time.time()
    print('ranking time = ', second_time - start_time)


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    testloader = load_dataloader(opt, split='test')

    dispatcher(opt, model, testloader, device)


if __name__ == '__main__':
    main()
