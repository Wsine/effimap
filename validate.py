import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import copy

from dataset import load_dataloader
from model import get_device, load_model
from arguments import parser
from utils import *


def get_mutate_filters(opt):
    df = pd.read_csv(get_output_location(opt, 'filter_assess.csv'), index_col=0)

    columns = [f'acc_c{c}' for c in range(opt.num_classes)]
    max_values = df[columns].max(axis=1)  # type: ignore
    max_columns = df[columns].idxmax(axis=1)  # type: ignore

    result = {}
    for c in range(opt.num_classes):
        column = f'acc_c{c}'
        df['margin'] = max_values - df[column]  # type: ignore
        sub_df = df[(df[column] > 0) & (max_columns != column)]  # type: ignore
        sub_df = sub_df.sort_values('margin', ascending=False)
        sub_df = sub_df.head(opt.batch_size)

        mutate_layers, accu_idx = {}, 0
        for layer in list(set(sub_df['layer'].tolist())):
            mutate_filters = sub_df[sub_df['layer'] == layer]['filter_idx'].tolist()
            mutate_layers[layer] = {
                'start_idx': accu_idx,
                'filters': mutate_filters
            }
            accu_idx += len(mutate_filters)

        result[f'c{c}'] = mutate_layers

    return result


@torch.no_grad()
def input_validate(opt, model, loader, device):
    model2 = copy.deepcopy(model)
    model.eval()
    model2.eval()
    mutate_info = get_mutate_filters(opt)

    def _mask_out_channel(pred, layer):
        mutate_layers = mutate_info[f'c{pred}']
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
    consist = [0, 0, 0, 0]
    for inputs, targets in tqdm(loader, desc='Validate'):
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

        exp_inputs = inputs.repeat(opt.batch_size, 1, 1, 1)
        outputs = model2(exp_inputs)
        _, predicted2 = outputs.max(1)
        if predicted2.eq(predicted).sum().item() == opt.batch_size:
            if predicted.item() == targets.item():
                consist[0] += 1
            else:
                consist[1] += 1
        else:
            if predicted.item() == targets.item():
                consist[2] += 1
            else:
                consist[3] += 1

    acc = 100. * correct / total
    print('test acc: {:.2f}%'.format(acc))

    print('validate acc 0: {:.2f}%'.format(100. * consist[0] / total))
    print('validate acc 1: {:.2f}%'.format(100. * consist[1] / total))
    print('validate acc 2: {:.2f}%'.format(100. * consist[2] / total))
    print('validate acc 3: {:.2f}%'.format(100. * consist[3] / total))

    print('validate acc: {:.2f}%'.format(100. * (consist[0] + consist[3] ) / total))



def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)

    b, opt.batch_size = opt.batch_size, 1
    testloader = load_dataloader(opt, split='test')
    opt.batch_size = b

    input_validate(opt, model, testloader, device)
    #  result = input_validate(opt, model, testloader, device)
    #  result_name = 'filter_assess.csv'
    #  export_object(opt, result_name, result)


if __name__ == '__main__':
    main()

