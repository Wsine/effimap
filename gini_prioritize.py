import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import parser
from model import load_model
from dataset import load_dataloader
from metric import post_predict, correctness
from utils import get_device, guard_folder, save_object


@torch.no_grad()
def prioritize_with_deepgini(ctx):
    device = get_device(ctx)
    testloader = load_dataloader(ctx, split='test')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()

    gini_impurity, oracle = [], []
    for inputs, targets in tqdm(testloader, desc='Gini'):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_preds = post_predict(ctx, model(inputs))

        _, inputs_probs = inputs_preds
        gini = F.softmax(inputs_probs, dim=1).square().sum(dim=1).mul(-1.).add(1.)
        gini_impurity.append(gini)

        incorrect = correctness(ctx, inputs_preds, targets, invert=True)
        oracle.append(incorrect)
    gini_impurity = torch.cat(gini_impurity)
    oracle = torch.cat(oracle)

    _, indices = torch.sort(gini_impurity, descending=True)
    oracle = oracle[indices]

    result = {
        'rank': oracle.numpy(),
        'ideal': oracle.sort(descending=True).values.numpy(),
        'worst': oracle.sort(descending=False).values.numpy()
    }

    save_object(ctx, result, 'gini_list.pkl')


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)
    prioritize_with_deepgini(ctx)


if __name__ == '__main__':
    main()

