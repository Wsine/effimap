import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import pandas as pd
import numpy as np

from dataset import load_dataloader
from model import get_device, load_model
from models.vanilla_vae import VanillaVAE
from mutate import feature_hook, feature_container
from train import activation_hook, activation_trace
from arguments import parser
from utils import *


dispatcher = AttrDispatcher('prioritor')


def rauc_measurement(y):
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
                'actual': e.item()
            }, ignore_index=True)

    # statistic result
    print('test acc: {:.2f}%'.format(100. * correct / total))
    df = df.astype({'gini': float, 'actual': int})
    df.sort_values(by=['gini'], ascending=True, inplace=True)
    for r in [100, 200, 300, 500]:
        rauc = rauc_measurement(df.head(r)['actual'])
        print('rauc for gini {} samples: {:.2f}%'.format(r, 100. * rauc))
    for r in [0.1, 0.2, 0.3, 0.5]:
        sub_seq = int(df.size * r)
        rauc = rauc_measurement(df.head(sub_seq)['actual'])
        print('rauc for gini {} samples: {:.2f}%'.format(r, 100. * rauc))
    rauc = rauc_measurement(df['actual'])
    print('rauc for gini all: {:.2f}%'.format(100. * rauc))


@dispatcher.register('LSA')
@torch.no_grad()
def likelihood_based_surprise_adequacy(opt, model, dataloader, device):
    model.eval()
    if opt.dataset == 'cifar100' and opt.model == 'resnet32':
        model.layer1[4].bn1.register_forward_hook(activation_hook)
    elif opt.dataset == 'tinyimagenet' and opt.model == 'resnet18':
        model.layer3[0].bn1.register_forward_hook(activation_hook)
    elif opt.dataset == 'mnist' and opt.model == 'mlp':
        model.model.fc2.register_forward_hook(activation_hook)
    elif opt.dataset == 'svhn' and opt.model == 'svhn':
        model.features[5].register_forward_hook(activation_hook)

    lsa_model = load_object(opt, 'kernel_density_estimator.pkl')
    kde = lsa_model['kde']  # type: ignore
    keep_columns = lsa_model['keep_columns']  # type: ignore

    equals_pool, score_pool = [], []
    for inputs, targets in tqdm(dataloader, desc='Test'):
        outputs = model(inputs.to(device))
        if opt.task == 'regress':
            predicted = outputs.flatten()
            delta = predicted.view(-1) - targets.to(device).view(-1)
            equals_pool.append(delta.abs().cpu())
        else:
            _, predicted = outputs.max(1)
            equals = predicted.eq(targets.to(device)).cpu()
            equals_pool.append(equals)
        test_at = activation_trace[-1][:, keep_columns]
        scores = kde.score_samples(test_at)
        score_pool.append(scores)
    equals_pool = torch.cat(equals_pool)
    score_pool = np.concatenate(score_pool)
    print(score_pool.shape)

    # test_at = np.concatenate(activation_trace)[:, keep_columns]
    # print(test_at.shape)
    # # scores = kde.score_samples(test_at)
    # scores = -kde.logpdf(test_at)
    # print(scores.shape)
    # print(scores)

    df = pd.DataFrame()
    for s, e in zip(score_pool, equals_pool):
        df = df.append({
            'lsa': -1.0 * s,
            'actual': e.item()
        }, ignore_index=True)
    df = df.astype({'lsa': float, 'actual': int})
    # df = df.astype({'lsa': float, 'actual': float})
    df.sort_values(by=['lsa'], ascending=True, inplace=True)
    for r in [100, 200, 300, 500]:
        rauc = rauc_measurement(df.head(r)['actual'])
        print('rauc for LSA {} samples: {:.2f}%'.format(r, 100. * rauc))
    for r in [0.1, 0.2, 0.3, 0.5]:
        sub_seq = int(df.size * r)
        rauc = rauc_measurement(df.head(sub_seq)['actual'])
        print('rauc for LSA {:.2f}%: {:.2f}%'.format(r * 100, 100. * rauc))
    rauc = rauc_measurement(df['actual'])
    print('rauc for LSA all: {:.2f}%'.format(100. * rauc))




@dispatcher.register('furret')
@torch.no_grad()
def estimator_with_mutation_score(opt, model, dataloader, device):
    for module in model.modules():
        module.register_forward_hook(feature_hook)
    model.eval()

    mutation_model = load_object(opt, 'mutation_estimator.pkl')

    df = pd.DataFrame()
    correct, total = 0, 0
    features_pool, entropy_pool, equals_pool = [], [], []
    for inputs, targets in tqdm(dataloader, desc='Validate'):
        inputs = inputs.to(device)
        feature_container.clear()
        outputs = model(inputs)
        features = torch.stack(feature_container, dim=-1).cpu()
        features_pool.append(features)

        if opt.task == 'regress':
            entropy_pool.append(torch.zeros((inputs.size(0), )))  # useless
            predicted = outputs.flatten()
            delta = predicted.view(-1) - targets.to(device).view(-1)
            equals_pool.append(delta.abs().cpu())
            correct += delta.abs().sum().item()
        else:
            probs = F.softmax(outputs, dim=1)
            # entropy = probs.mul(torch.log(probs + 1e-8)).sum(dim=1).mul(-1.)  # shannon entropy
            # entropy = torch.log(probs.square().sum(dim=1)).mul(-1.)  # collision entropy
            entropy = torch.log(probs.max(dim=1).values).mul(-1.)  # min entropy
            entropy_pool.append(entropy.cpu())
            _, predicted = outputs.max(1)
            equals = predicted.eq(targets.to(device)).cpu()
            equals_pool.append(equals)
            correct += equals.sum().item()
        total += targets.size(0)

    features_pool = torch.cat(features_pool)
    entropy_pool = torch.cat(entropy_pool)
    equals_pool = torch.cat(equals_pool)
    mutations = mutation_model.predict(features_pool.numpy())  # type: ignore

    for m, en, e in zip(mutations, entropy_pool, equals_pool):
        df = df.append({
            # 'furret': m.sum() if opt.task == 'regress' else m.sum() / len(m) + g,
            'furret': m.sum() if opt.task == 'regress' else m.sum() / len(m) * en.item(),
            # 'furret': m.sum() if opt.task == 'regress' else m.sum() / len(m),
            # 'furret': m.sum() if opt.task == 'regress' else g,
            'actual': e.item() if opt.task == 'regress' else e.item()
        }, ignore_index=True)

    print('test perf: {:.2f}%'.format(100. * correct / total))
    df = df.astype({'furret': float, 'actual': int})
    df.sort_values(by=['furret'], ascending=True, inplace=True)
    # df = df.astype({'furret': float, 'actual': float})
    # df.sort_values(by=['furret'], ascending=False, inplace=True)
    for r in [100, 200, 300, 500]:
        rauc = rauc_measurement(df.head(r)['actual'])
        print('rauc for furret {} samples: {:.2f}%'.format(r, 100. * rauc))
    for r in [0.1, 0.2, 0.3, 0.5]:
        sub_seq = int(df.size * r)
        rauc = rauc_measurement(df.head(sub_seq)['actual'])
        print('rauc for furret {:.2f}%: {:.2f}%'.format(r * 100, 100. * rauc))
    rauc = rauc_measurement(df['actual'])
    print('rauc for furret all: {:.2f}%'.format(100. * rauc))


@dispatcher.register('furret2')
@torch.no_grad()
def estimator_with_multi_mutation_score(opt, model, dataloader, device):
    for module in model.modules():
        module.register_forward_hook(feature_hook)
    model.eval()

    sample_img = next(iter(dataloader))[0][0]
    img_channels, img_size = sample_img.size(0), sample_img.size(-1)
    if img_size < 32:
        pad = torchvision.transforms.Pad((32 - img_size) // 2)
        crop = torchvision.transforms.CenterCrop(img_size)
        img_size = 32
    else:
        pad = crop = None
    generator = VanillaVAE(img_channels, img_size, 10)
    state = load_object(opt, 'encoder_model.pt')
    generator.load_state_dict(state['net'])  # type: ignore
    generator = generator.to(device)
    generator.eval()

    mutation_model = load_object(opt, 'mutation_estimator.pkl')

    df = pd.DataFrame()
    features_pool, entropy_pool, delta_pool = [], [], []
    for inputs, targets in tqdm(dataloader, desc='Validate'):
        inputs, targets = inputs.to(device), targets.to(device)
        feature_container.clear()
        outputs = model(inputs)
        features = torch.stack(feature_container, dim=-1).cpu()
        features_pool.append(features)
        predicted = outputs.flatten()
        delta = predicted.view(-1) - targets.view(-1)
        delta_pool.append(delta.abs().cpu())

        for input in tqdm(inputs, desc='Samples', leave=False):
            input_repeat = input.view(1, *input.size()) \
                    .repeat(opt.num_input_mutants, 1, 1, 1)
            if pad is not None:
                input_repeat = pad(input_repeat)
            input_mutants = generator.generate(input_repeat)
            if crop is not None:
                input_mutants = crop(input_mutants)
            feature_container.clear()
            input_mutants = input_mutants.reshape((opt.num_input_mutants, -1))  # for mnist only
            outputs = model(input_mutants).flatten()
            probs = F.softmax(outputs, dim=0)
            # entropy = torch.log(torch.tensor(input_mutants.size(0)))  # hartley entropy
            entropy = probs.mul(torch.log(probs)).sum().mul(-1.)  # shannon entropy
            entropy_pool.append(entropy)

    features_pool = torch.cat(features_pool)
    entropy_pool = torch.Tensor(entropy_pool)
    delta_pool = torch.cat(delta_pool)

    mutations = mutation_model.predict(features_pool.numpy())  # type: ignore
    for m, en, d in zip(mutations, entropy_pool, delta_pool):
        df = df.append({
            'furret': m.sum() / len(m) * en.item(),
            'actual': d.item()
        }, ignore_index=True)

    df = df.astype({'furret': float, 'actual': int})
    df.sort_values(by=['furret'], ascending=False, inplace=True)
    for r in [100, 200, 300, 500]:
        rauc = rauc_measurement(df.head(r)['actual'])
        print('rauc for furret {} samples: {:.2f}%'.format(r, 100. * rauc))
    for r in [0.1, 0.2, 0.3, 0.5]:
        sub_seq = int(df.size * r)
        rauc = rauc_measurement(df.head(sub_seq)['actual'])
        print('rauc for furret {:.2f}%: {:.2f}%'.format(r * 100, 100. * rauc))
    rauc = rauc_measurement(df['actual'])
    print('rauc for furret all: {:.2f}%'.format(100. * rauc))


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
    rank = ranking_model.predict(X)  # type: ignore

    sort_inds = rank.argsort()
    Y = Y[sort_inds]
    for r in [100, 200, 300, 500]:
        rauc = rauc_measurement(Y[:r])
        print('rauc for prima {} samples: {:.2f}%'.format(r, 100. * rauc))
    for r in [0.1, 0.2, 0.3, 0.5]:
        sub_seq = int(len(Y) * r)
        rauc = rauc_measurement(Y[:sub_seq])
        print('rauc for prima {:.2f}%: {:.2f}%'.format(r * 100, 100. * rauc))
    rauc = rauc_measurement(Y)
    print('rauc for prima all: {:.2f}%'.format(100. * rauc))


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

        equals = predicted.eq(targets.to(device))
        for p, e in zip(pvscore, equals):
            df = df.append({
                'dissector': p.item(),
                'actual': e.item()
            }, ignore_index=True)

    df = df.astype({'dissector': float, 'actual': int})
    df.sort_values(by=['dissector'], ascending=False, inplace=True)
    for r in [100, 200, 300, 500]:
        rauc = rauc_measurement(df.head(r)['actual'])
        print('rauc for dissector {} samples: {:.2f}%'.format(r, 100. * rauc))
    for r in [0.1, 0.2, 0.3, 0.5]:
        sub_seq = int(df.size * r)
        rauc = rauc_measurement(df.head(sub_seq)['actual'])
        print('rauc for dissector {} samples: {:.2f}%'.format(r, 100. * rauc))
    rauc = rauc_measurement(df['actual'])
    print('rauc for dissector all: {:.2f}%'.format(100. * rauc))


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    testloader = load_dataloader(opt, split='test')

    dispatcher(opt, model, testloader, device)


if __name__ == '__main__':
    main()
