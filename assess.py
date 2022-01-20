import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

from dataset import load_dataloader
from model import get_device, load_model
from arguments import parser
from utils import *


@torch.no_grad()
def evaluate_accuracy(opt, model, dataloader, device):
    model.eval()

    correct, total = 0, 0
    confusion_matrix = torch.zeros(opt.num_classes, opt.num_classes)
    for inputs, targets in tqdm(dataloader, desc='Evaluate', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        for t, p in zip(targets, predicted):
            confusion_matrix[t.item(), p.item()] += 1

    acc = 100. * correct / total
    class_acc = (confusion_matrix.diag() / confusion_matrix.sum(1)).tolist()
    return acc, class_acc


def performance_difference(opt, model, valloader, device):
    base_acc, base_c_acc = evaluate_accuracy(opt, model, valloader, device)
    print('base acc =', base_acc)
    print('base class acc =', base_c_acc)

    def _mask_out_channel(chn):
        def __hook(module, finput, foutput):
            foutput[:, chn] = 0
            return foutput
        return __hook

    df = pd.DataFrame(columns=['layer', 'filter_idx', 'acc'] + \
                      [f'acc_c{c}' for c in range(opt.num_classes)])
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for lname in tqdm(conv_names, desc='Modules', leave=True):
        module = rgetattr(model, lname)
        for chn in tqdm(range(module.out_channels), desc='Filters', leave=False):
            handle = module.register_forward_hook(_mask_out_channel(chn))
            acc, c_acc = evaluate_accuracy(opt, model, valloader, device)
            r1 = { 'layer': lname, 'filter_idx': chn, 'acc': acc - base_acc }
            r2 = { f'acc_c{c}': c_acc[c] - base_c_acc[c] for c in range(opt.num_classes) }
            df = df.append({**r1, **r2}, ignore_index=True)
            handle.remove()

    return df


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    valloader = load_dataloader(opt, split='val')

    result = performance_difference(opt, model, valloader, device)
    result_name = 'filter_assess.csv'
    export_object(opt, result_name, result)


if __name__ == '__main__':
    main()

