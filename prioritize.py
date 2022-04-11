import torch
import torch.nn as nn
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


@dispatcher.register('estimate')
@torch.no_grad()
def estimator_method(opt, model, dataloader, device):
    for module in model.modules():
        module.register_forward_hook(feature_hook)
    model.eval()

    mutation_model = load_object(opt, 'mutation_estimator.pkl')
    #  ranking_model = load_object(opt, 'ranking_model.pkl')

    df = pd.DataFrame()
    correct, total = 0, 0
    for inputs, targets in tqdm(dataloader, desc='Validate'):
        inputs, targets = inputs.to(device), targets.to(device)
        feature_container.clear()
        outputs = model(inputs)
        features = torch.stack(feature_container, dim=-1).cpu().numpy()
        mutation = mutation_model.predict(features)  # type: ignore
        #  ranking = ranking_model.predict_proba(mutation)  # type: ignore

        _, predicted = outputs.max(1)
        equals = predicted.eq(targets)
        correct += equals.sum().item()
        total += targets.size(0)

        for m, e in zip(mutation, equals):
            df = df.append({
                'mutation_score': m.sum(),
                'actual': (~e).item()
            }, ignore_index=True)
        #  for m, r, e in zip(mutation, ranking, equals):
        #      df = df.append({
        #          'mutation_score': m.sum(),
        #          'ranking': r[1],
        #          'actual': e.item()
        #      }, ignore_index=True)

    # statistic result
    print('test acc: {:.2f}%'.format(100. * correct / total))

    #  df = df.astype({'mutation_score': float, 'ranking': float, 'actual': int})
    df = df.astype({'mutation_score': float, 'actual': int})
    fpr, tpr, _ = metrics.roc_curve(df['actual'], df['mutation_score'])
    auc = metrics.auc(fpr, tpr)
    print('auc for mutation score: {:.2f}%'.format(100. * auc))
    #  fpr, tpr, _ = metrics.roc_curve(df['actual'], df['ranking'])
    #  auc = metrics.auc(fpr, tpr)
    #  print('auc for ranking model: {:.2f}%'.format(100. * auc))


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    testloader = load_dataloader(opt, split='test')

    dispatcher(opt, model, testloader, device)


if __name__ == '__main__':
    main()
