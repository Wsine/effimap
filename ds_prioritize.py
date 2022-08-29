import torch
from tqdm import tqdm

from arguments import parser
from model import load_model
from dataset import load_dataloader
from metric import post_predict, correctness
from utils import get_device, guard_folder, load_torch_object, save_object


@torch.no_grad()
def prioritize_with_dissector(ctx):
    device = get_device(ctx)
    testloader = load_dataloader(ctx, split='test')
    model = load_model(ctx, pretrained=True).to(device)
    model.eval()

    if ctx.dataset == 'cifar100' and ctx.model == 'resnet32':
        layers = ['relu', 'layer1', 'layer2', 'layer3']
    elif ctx.dataset == 'tinyimagenet' and ctx.model == 'resnet18':
        layers = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
    else:
        raise ValueError('Not supported combinations now')

    submodels = []
    for name in layers:
        submodel = load_torch_object(ctx, f'snapshotors/snapshotor_{name}.pt')
        assert(submodel is not None)
        submodel = submodel.to(device)
        submodel.eval()
        submodels.append(submodel)

    pvscores, oracle = [], []
    for inputs, targets in tqdm(testloader, desc='Dissector'):
        inputs, targets = inputs.to(device), targets.to(device)
        model_preds = post_predict(ctx, model(inputs))
        model_labels, _ = model_preds

        snap_preds = []
        for submodel in submodels:
            submodel_preds = post_predict(ctx, submodel(inputs))
            snap_preds.append(submodel_preds)

        snapshot = torch.stack([
            snap_probs for _, snap_probs in snap_preds
        ], dim=-1)
        highest = snapshot.max(1).values
        second_highest = snapshot.topk(2, dim=1).values[:, 1, :]
        model_high = torch.stack([snapshot[i, p, :] for i, p in enumerate(model_labels)])
        svscore1 = highest / (highest + second_highest)
        svscore2 = 1 - highest / (model_high + highest)
        mask = snapshot.max(1).indices == \
               model_labels.repeat(snapshot.size(-1), 1).transpose(0, 1)
        svscore = torch.where(mask, svscore1, svscore2)
        weights = torch.log(torch.arange(1, svscore.size(1) + 1, device=device))
        pvscore = (svscore * weights).sum(dim=1) / weights.sum()
        pvscores.append(pvscore)

        incorrect = correctness(ctx, model_preds, targets, invert=True)
        oracle.append(incorrect)

    pvscores = torch.cat(pvscores)
    oracle = torch.cat(oracle)

    _, indices = torch.sort(pvscores, descending=False)
    oracle = oracle[indices]

    result = {
        'rank': oracle.numpy(),
        'ideal': oracle.sort(descending=True).values.numpy(),
        'worst': oracle.sort(descending=False).values.numpy()
    }

    save_object(ctx, result, 'ds_list.pkl')


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)
    prioritize_with_dissector(ctx)


if __name__ == '__main__':
    main()

