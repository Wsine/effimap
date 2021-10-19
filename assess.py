import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_dataloader
from model import get_device, load_model
from arguments import parser
from utils import *


@torch.no_grad()
def evaluate_accuracy(opt, model, dataloader, device):
    model.eval()

    confusion_matrix = torch.zeros(opt.num_classes, opt.num_classes)
    for inputs, targets in tqdm(dataloader, desc='Evaluate', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        _, predicted = outputs.max(1)
        for t, p in zip(targets, predicted):
            confusion_matrix[t.item(), p.item()] += 1

    class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    return class_acc


def performance_improve(opt, model, valloader, device):
    base_c_acc = evaluate_accuracy(opt, model, valloader, device)
    print('base class acc =', base_c_acc)

    def _mask_out_channel(chn):
        def __hook(module, finput, foutput):
            foutput[:, chn] = 0
            return foutput
        return __hook

    suspicious = { f'class{c}': {} for c in range(opt.num_classes) }
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for lname in tqdm(conv_names, desc='Modules', leave=True):
        module = rgetattr(model, lname)
        indices = [[] for _ in range(opt.num_classes)]
        for chn in tqdm(range(module.out_channels), desc='Filters', leave=False):
            handle = module.register_forward_hook(_mask_out_channel(chn))
            c_acc = evaluate_accuracy(opt, model, valloader, device)
            for c in range(opt.num_classes):
                if c_acc[c] > base_c_acc[c]:
                    indices[c].append(chn)
            handle.remove()
        for c in range(opt.num_classes):
            suspicious[f'class{c}'][lname] = indices[c]

    return suspicious


@torch.no_grad()
def evaluate_probability(opt, model, dataloader, device):
    model.eval()

    class_prob = [[] for _ in range(opt.num_classes)]
    for inputs, targets in tqdm(dataloader, desc='Evaluate', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        for p, t in zip(probs, targets):
            class_prob[t.item()].append(p[t.item()])

    avg_class_prob = torch.tensor([torch.tensor(probs).mean() for probs in class_prob])
    return avg_class_prob


def probability_increase(opt, model, valloader, device):
    base_c_prob = evaluate_probability(opt, model, valloader, device)
    print('base class prob =', base_c_prob)

    def _mask_out_channel(chn):
        def __hook(module, finput, foutput):
            foutput[:, chn] = 0
            return foutput
        return __hook

    suspicious = { f'class{c}': {} for c in range(opt.num_classes) }
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for lname in tqdm(conv_names, desc='Modules', leave=True):
        module = rgetattr(model, lname)
        indices = [[] for _ in range(opt.num_classes)]
        for chn in tqdm(range(module.out_channels), desc='Filters', leave=False):
            handle = module.register_forward_hook(_mask_out_channel(chn))
            c_prob = evaluate_probability(opt, model, valloader, device)
            for c in range(opt.num_classes):
                if c_prob[c] > base_c_prob[c]:
                    indices[c].append(chn)
            handle.remove()
        for c in range(opt.num_classes):
            suspicious[f'class{c}'][lname] = indices[c]

    return suspicious


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    valloader = load_dataloader(opt, split='val')

    result = probability_increase(opt, model, valloader, device)
    result_name = 'susp_filters.json'
    export_object(opt, result_name, result)


if __name__ == '__main__':
    main()

