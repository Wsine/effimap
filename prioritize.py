import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn import metrics

from dataset import load_dataloader
from model import get_device, load_model
from extract import feature_hook
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


feature_container = []
def feature_hook(module, inputs, outputs):
    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
        mean = torch.mean(outputs, dim=(1, 2, 3))
        feature_container.append(mean)
        var = torch.var(outputs, dim=(1, 2, 3))
        feature_container.append(var)
    if isinstance(module, nn.ReLU):
        in_mask = (inputs[0] < 0).sum(dim=(1, 2, 3)) / outputs[0].numel()
        out_mask = (outputs < 0).sum(dim=(1, 2, 3)) / outputs[0].numel()
        feature_container.append(out_mask - in_mask)
    elif isinstance(module, nn.Linear):
        mean = torch.mean(inputs[0], dim=1)
        feature_container.append(mean)
        var = torch.var(inputs[0], dim=1)
        feature_container.append(var)
        gini = F.softmax(outputs, dim=1).square().sum(dim=1).mul(-1.).add(1.)
        feature_container.append(gini)


@dispatcher.register('estimate')
@torch.no_grad()
def estimator_method(opt, model, dataloader, device):
    for module in model.modules():
        module.register_forward_hook(feature_hook)
    model.eval()

    mutation_model = load_object(opt, 'mutation_estimator.pkl')
    ranking_model = load_object(opt, 'ranking_model.pkl')

    df = pd.DataFrame()
    correct, total = 0, 0
    for inputs, targets in tqdm(dataloader, desc='Validate'):
        inputs, targets = inputs.to(device), targets.to(device)
        feature_container.clear()
        outputs = model(inputs)
        features = torch.stack(feature_container, dim=-1).cpu().numpy()
        mutation = mutation_model.predict(features)  # type: ignore
        ranking = ranking_model.predict_proba(mutation)  # type: ignore

        _, predicted = outputs.max(1)
        equals = predicted.eq(targets)
        correct += equals.sum().item()
        total += targets.size(0)

        for m, r, e in zip(mutation, ranking, equals):
            df = df.append({
                'mutation_score': m.sum(),
                'ranking': r[1],
                'actual': e.item()
            }, ignore_index=True)

    # statistic result
    print('test acc: {:.2f}%'.format(100. * correct / total))

    df = df.astype({'mutation_score': float, 'ranking': float, 'actual': int})
    fpr, tpr, _ = metrics.roc_curve(df['actual'], df['mutation_score'])
    auc = metrics.auc(fpr, tpr)
    print('auc for mutation score: {:.2f}%'.format(100. * auc))
    fpr, tpr, _ = metrics.roc_curve(df['actual'], df['ranking'])
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
