import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn import metrics

from dataset import load_dataloader
from model import get_device, load_model
from mutate import feature_hook, feature_container
from arguments import parser
from utils import *


dispatcher = AttrDispatcher('prioritor')


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
def estimator_method(opt, model, dataloader, device):
    for module in model.modules():
        module.register_forward_hook(feature_hook)
    model.eval()

    mutation_model = load_object(opt, 'mutation_estimator.pkl')

    df = pd.DataFrame()
    correct, total = 0, 0
    for inputs, targets in tqdm(dataloader, desc='Validate'):
        inputs, targets = inputs.to(device), targets.to(device)
        feature_container.clear()
        outputs = model(inputs)
        features = torch.stack(feature_container, dim=-1).cpu().numpy()
        mutation = mutation_model.predict(features)  # type: ignore

        _, predicted = outputs.max(1)
        equals = predicted.eq(targets)
        correct += equals.sum().item()
        total += targets.size(0)

        for m, e in zip(mutation, equals):
            df = df.append({
                'mutation_score': m.sum(),
                'actual': (~e).item()
            }, ignore_index=True)

    # statistic result
    print('test acc: {:.2f}%'.format(100. * correct / total))

    df = df.astype({'mutation_score': float, 'actual': int})
    fpr, tpr, _ = metrics.roc_curve(df['actual'], df['mutation_score'])
    auc = metrics.auc(fpr, tpr)
    print('auc for mutation score: {:.2f}%'.format(100. * auc))


@dispatcher.register('prima')
def prima_method(opt, model, dataloader, device):
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

    fpr, tpr, _ = metrics.roc_curve(Y, rank)
    auc = metrics.auc(fpr, tpr)
    print('auc for ranking model: {:.2f}%'.format(100. * auc))


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    testloader = load_dataloader(opt, split='test')

    dispatcher(opt, model, testloader, device)


if __name__ == '__main__':
    main()
