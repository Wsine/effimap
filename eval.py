import json

import torch
import torch.utils.data
import scipy.integrate
from tqdm import tqdm

from dataset import load_dataloader
from model import get_device, load_model
from arguments import parser
from utils import *


class PrioritizedSampler(torch.utils.data.Sampler[int]):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    accu_failed, accu_sum = [], 0
    correct, total = 0, 0
    with tqdm(dataloader, desc='Evaluate') as tepoch:
        for inputs, targets in tepoch:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            cum = predicted.ne(targets).flatten().cumsum(dim=0).add(accu_sum).cpu()
            accu_failed.append(cum)
            accu_sum = cum[-1].item()

            acc = 100. * correct / total
            tepoch.set_postfix(acc=acc)

    acc = 100. * correct / total
    accu_failed = torch.cat(accu_failed).tolist()

    return acc, accu_failed


def compute_area_under_curve(y):
    ideal_y = (list(range(y[-1])) + [y[-1]] * len(y))[:len(y)]
    area_y = scipy.integrate.simps(y)
    area_iy = scipy.integrate.simps(ideal_y)
    ratio = 100. * area_y / area_iy
    return ratio


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    if opt.metric == 'none':
        print('please provide a metric to evaluate')
        return

    device = get_device(opt)
    model = load_model(opt).to(device)

    if opt.metric == 'rauc':
        with open(get_output_location(opt, 'mutation_kill_priority.json')) as f:
            priority = json.load(f)
        sample_order = [i for i, _ in priority]
        sampler = PrioritizedSampler(sample_order)
    else:
        sampler = None

    testloader = load_dataloader(opt, split='test', sampler=sampler)
    acc, accu_failed = evaluate(model, testloader, device)

    if opt.metric == 'acc':
        print('base acc: {:.2f}%'.format(acc))
    elif opt.metric == 'rauc':
        rauc = compute_area_under_curve(accu_failed)
        print('rauc of current options: {:.2f}%'.format(rauc))
    else:
        raise ValueError('Invalid metric')


if __name__ == "__main__":
    main()

