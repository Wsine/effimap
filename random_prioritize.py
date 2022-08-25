import torch
from tqdm import tqdm

from arguments import parser
from model import load_model
from dataset import load_dataloader
from metric import correctness
from utils import get_device, guard_folder, save_object


@torch.no_grad()
def prioritize_randomly(ctx):
    device = get_device(ctx)
    testloader = load_dataloader(ctx, split='test')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()

    oracle = []
    for inputs, targets in tqdm(testloader, desc='Random'):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        incorrect = correctness(ctx, predicted, targets, invert=True)
        oracle.append(incorrect)
    oracle = torch.cat(oracle)

    result = {}
    for i in range(100):
        result[f'rank{i}'] = oracle[torch.randperm(oracle.size(0))].numpy()
    result['ideal'] = oracle.sort(descending=True).values.numpy()
    result['worst'] = oracle.sort(descending=False).values.numpy()

    save_object(ctx, result, 'random_list.pkl')


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)
    prioritize_randomly(ctx)


if __name__ == '__main__':
    main()

