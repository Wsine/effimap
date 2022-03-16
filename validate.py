import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import copy
from sklearn import metrics

from dataset import load_dataloader
from model import get_device, load_model
from arguments import parser
from utils import *


dispatcher = AttrDispatcher('validate')


def get_mutate_filters(opt):
    df = pd.read_csv(get_output_location(opt, 'filter_assess.csv'), index_col=0)

    #  columns = [f'acc_c{c}' for c in range(opt.num_classes)]
    #  max_values = df[columns].max(axis=1)  # type: ignore
    #  max_columns = df[columns].idxmax(axis=1)  # type: ignore

    result = {}
    for c in range(opt.num_classes):
        column = f'acc_c{c}'
        #  df['margin'] = max_values - df[column]  # type: ignore
        #  sub_df = df[(df[column] > 0) & (max_columns != column)]  # type: ignore
        #  sub_df = sub_df.sort_values('margin', ascending=False)
        sub_df = df[(df[column] > -1e-8) & (df[column] < 1e-2) & (df['acc'] > 0)]  # type: ignore
        sub_df = sub_df.sort_values('acc', ascending=False)
        sub_df = sub_df.head(opt.batch_size)

        mutate_layers, accu_idx = {}, 0
        for layer in list(set(sub_df['layer'].tolist())):
            mutate_filters = sub_df[sub_df['layer'] == layer]['filter_idx'].tolist()
            mutate_layers[layer] = {
                'start_idx': accu_idx,
                'filters': mutate_filters
            }
            accu_idx += len(mutate_filters)

        result[f'c{c}'] = {
            'layers': mutate_layers,
            'num': accu_idx
        }

    return result


@dispatcher.register('mutants_killed')
@torch.no_grad()
def mutants_killed(opt, model, device):
    b, opt.batch_size = opt.batch_size, 1
    testloader = load_dataloader(opt, split='test')
    opt.batch_size = b

    model2 = copy.deepcopy(model)
    model.eval()
    model2.eval()
    mutate_info = get_mutate_filters(opt)

    def _mask_out_channel(pred, layer):
        mutate_layers = mutate_info[f'c{pred}']['layers']
        if layer not in mutate_layers.keys():
            def __hook1(module, finput, foutput):
                pass
            return __hook1
        else:
            mutate_filters = mutate_layers[layer]
            def __hook2(module, finput, foutput):
                start_idx = mutate_filters['start_idx']
                for i, f in enumerate(mutate_filters['filters']):
                    foutput[start_idx + i, f] = 0
                return foutput
            return __hook2


    hooks, prev_pred = [], -1
    conv_names = [n for n, m in model2.named_modules() if isinstance(m, nn.Conv2d)]
    correct, total = 0, 0
    tp, fp, fn, tn = 0, 0, 0, 0
    df = pd.DataFrame()
    for inputs, targets in tqdm(testloader, desc='Validate'):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        if predicted.item() == targets.item():
            correct += 1
        total += targets.size(0)

        if predicted.item() != prev_pred:
            for h in hooks: h.remove()
            hooks.clear()
            for lname in conv_names:
                module = rgetattr(model2, lname)
                handle = module.register_forward_hook(
                    _mask_out_channel(predicted.item(), lname))
                hooks.append(handle)
            prev_pred = predicted.item()

        num_mutants = mutate_info[f'c{predicted.item()}']['num']
        exp_inputs = inputs.repeat(num_mutants, 1, 1, 1)
        exp_outputs = model2(exp_inputs)
        _, exp_predicted = exp_outputs.max(1)

        batch_validate_sum = exp_predicted.eq(predicted).sum()
        if predicted.item() == targets.item():
            if batch_validate_sum.item() == num_mutants:
                tp += 1
            else:
                fp += 1
        else:
            if batch_validate_sum.item() == num_mutants:
                fn += 1
            else:
                tn += 1

        gini = F.softmax(outputs, dim=1).square().sum(dim=1).mul(-1.).add(1.)
        df = df.append({
            'gini': gini,
            'change_rate': (num_mutants - batch_validate_sum.item()) / num_mutants,
            'validate': num_mutants - batch_validate_sum.item(),
            'actual': predicted.item() != targets.item()
        }, ignore_index=True)

    # statistic result
    print('test acc: {:.2f}%'.format(100. * correct / total))

    print('validate tp: {:.2f}%'.format(100. * tp / total))
    print('validate fp: {:.2f}%'.format(100. * fp / total))
    print('validate fn: {:.2f}%'.format(100. * fn / total))
    print('validate tn: {:.2f}%'.format(100. * tn / total))
    print('validate tpr: {:.2f}%'.format(100. * tp / (tp + fn)))
    print('validate fpr: {:.2f}%'.format(100. * fp / (fp + tn)))

    df = df.astype({'gini': float, 'change_rate': float, 'validate': float, 'actual': int})

    fpr, tpr, _ = metrics.roc_curve(df['actual'], df['validate'])
    auc = metrics.auc(fpr, tpr)
    print('validate auc for single: {:.2f}%'.format(100. * auc))

    fpr, tpr, _ = metrics.roc_curve(df['actual'], df['change_rate'])
    auc = metrics.auc(fpr, tpr)
    print('validate auc for change rate: {:.2f}%'.format(100. * auc))

    fpr, tpr, _ = metrics.roc_curve(df['actual'], df['gini'])
    auc = metrics.auc(fpr, tpr)
    print('validate auc for gini: {:.2f}%'.format(100. * auc))


@dispatcher.register('bntrend')
@torch.no_grad()
def bn_trend(opt, model, device):
    model.eval()

    bn_running_stats = torch.load(get_output_location(opt, 'bn_running_stats.pt'))

    bn_input = {}
    def _guard_bn_trend(name):
        def __hook(module, inputs):
            bn_input[name] = inputs[0]
        return __hook

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.register_forward_pre_hook(_guard_bn_trend(name))

    def _intermediate_bn_sum(finput, mean, var, momentum, eps):
        bn = F.batch_norm(
            #  finput, mean, var,
            finput, mean, torch.ones_like(mean),
            None, None,  # weight, bias
            False,  # bn_training
            0.0 if momentum is None else momentum,
            eps
        )
        return bn.sum(dim=(1, 2, 3))

    correct, total = 0, 0
    tp, fp, fn, tn = 0, 0, 0, 0
    df = pd.DataFrame()
    for clx_idx in tqdm(range(opt.num_classes), desc='Class', leave=True):
        testloader = load_dataloader(opt, split='test', single_class=clx_idx, download=False)
        bn_stats = bn_running_stats[f'c{clx_idx}']

        for inputs, targets in tqdm(testloader, desc='Data', leave=False):
            bn_input.clear()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            equal = predicted.eq(targets)
            correct += equal.sum().item()
            total += targets.size(0)

            OOD = torch.zeros((targets.size(0),), dtype=torch.int, device=device)
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    finput, bn_stat = bn_input[name], bn_stats[name]
                    ci_mean = bn_stat['running_mean'].to(device)
                    ci_var = bn_stat['running_var'].to(device)
                    ci_bn = _intermediate_bn_sum(
                        finput, ci_mean, ci_var,
                        module.momentum, module.eps
                    )
                    cur_bn = _intermediate_bn_sum(
                        finput, module.running_mean, module.running_var,
                        module.momentum, module.eps
                    )
                    OOD += (ci_bn > cur_bn).int()

            tp += torch.logical_and(equal, OOD.bool().logical_not()).sum().item()
            fp += torch.logical_and(equal, OOD).sum().item()
            fn += torch.logical_and(equal.logical_not(), OOD.bool().logical_not()).sum().item()
            tn += torch.logical_and(equal.logical_not(), OOD).sum().item()

            for e, o in zip(equal, OOD):
                df = df.append({
                    'bn': o.item(),
                    'actual': e.item()
                }, ignore_index=True)

    # statistic result
    print('test acc: {:.2f}%'.format(100. * correct / total))

    print('validate tp: {:.2f}%'.format(100. * tp / total))
    print('validate fp: {:.2f}%'.format(100. * fp / total))
    print('validate fn: {:.2f}%'.format(100. * fn / total))
    print('validate tn: {:.2f}%'.format(100. * tn / total))
    if tp + fn > 0:
        print('validate tpr: {:.2f}%'.format(100. * tp / (tp + fn)))
    if fp + fn > 0:
        print('validate fpr: {:.2f}%'.format(100. * fp / (fp + tn)))

    df = df.astype({'bn': int, 'actual': int})

    fpr, tpr, _ = metrics.roc_curve(df['actual'], df['bn'])
    auc = metrics.auc(fpr, tpr)
    print('validate auc for bn: {:.2f}%'.format(100. * auc))


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)
    guard_folder(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)

    dispatcher(opt, model, device)


if __name__ == '__main__':
    main()