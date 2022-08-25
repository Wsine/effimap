import torch
from tqdm import tqdm

from arguments import parser
from model import load_model
from dataset import load_dataloader
from utils import get_device


@torch.no_grad()
def evaluate_standard_accuracy(ctx):
    device = get_device(ctx)
    testloader = load_dataloader(ctx, split='val+test', download=True)
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()

    correct, total = 0, 0
    for inputs, targets in (pbar := tqdm(testloader, desc='Eval')):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        acc = 100. * correct / total
        pbar.set_postfix(acc=acc)

    acc = 100. * correct / total
    print('standard accuracy is {:.4f}%'.format(acc))


def main():
    ctx = parser.parse_args()
    print(ctx)
    evaluate_standard_accuracy(ctx)


if __name__ == '__main__':
    main()
