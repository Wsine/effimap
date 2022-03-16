import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from dataset import load_dataloader
from model import get_device, load_model
from arguments import parser
from utils import *


dispatcher = AttrDispatcher('assess')

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

    acc = correct / total
    class_acc = (confusion_matrix.diag() / confusion_matrix.sum(1)).tolist()
    return acc, class_acc


@dispatcher.register('perfdiff')
def performance_difference(opt, model, device):
    valloader = load_dataloader(opt, split='val')
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

    return df, 'filter_assess.csv'


@dispatcher.register('bnrunning')
@torch.no_grad()
def bn_running_mean_std(opt, model, device):
    def _bn_hook(module, inputs, outputs):
        return F.batch_norm(
            inputs[0],
            module.freeze_running_mean, module.freeze_running_var,
            module.weight, module.bias,
            False,  # bn_training
            0.0 if module.momentum is None else module.momentum,
            module.eps
        )

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            module.freeze_running_mean = module.running_mean.clone()  # type: ignore
            module.freeze_running_var = module.running_var.clone()  # type: ignore
            module.register_forward_hook(_bn_hook)  # type: ignore

    result = {}
    for clx_idx in range(opt.num_classes):
        print('Processing class {}'.format(clx_idx))
        trainloader = load_dataloader(opt, split='train', single_class=clx_idx)

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()

        for _ in tqdm(range(opt.epochs), desc='Epoch', leave=True):
            for inputs, targets in tqdm(trainloader, desc='BN', leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                _ = model(inputs)

        result[f'c{clx_idx}'] = {
            name: {
                'running_mean': module.running_mean.cpu(),
                'running_var': module.running_var.cpu()
            }
            for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)
        }

    return result, 'bn_running_stats.pt'


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)
    guard_folder(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)

    result, result_name = dispatcher(opt, model, device)
    export_object(opt, result_name, result)


if __name__ == '__main__':
    main()

