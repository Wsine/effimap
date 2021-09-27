import torch
from tqdm import tqdm

from dataset import load_dataloader
from model import get_device, load_model
from arguments import parser
from utils import *


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    correct, total = 0, 0
    with tqdm(dataloader, desc='Evaluate') as tepoch:
        for inputs, targets in tepoch:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            tepoch.set_postfix(acc=acc)

    acc = 100. * correct / total
    return acc


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    testloader = load_dataloader(opt, split='test')

    base_acc = evaluate(model, testloader, device)
    print('base_acc: {:.2f}%'.format(base_acc))


if __name__ == "__main__":
    main()

