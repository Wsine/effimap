import json

import torch
from tqdm import tqdm

from dataset import load_dataloader
from model import get_device, load_model
from arguments import parser
from utils import *


@torch.no_grad()
def evaluate(opt, model, valloader, device):
    model.eval()

    confusion_matrix = torch.zeros(opt.num_classes, opt.num_classes)
    for inputs, targets in tqdm(valloader, desc='Evaluate', leave=True):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        _, predicted = outputs.max(1)
        for p, t in zip(predicted, targets):
            confusion_matrix[p.item(), t.item()] += 1

    return confusion_matrix


@torch.no_grad()
def mutation_analyze(model, testloader, device, susp, cfsion_mat):
    model.eval()
    cfsion_mat = cfsion_mat.to(device)

    def _mask_out_channel(chn):
        def __hook(module, finput, foutput):
            foutput[:, chn] = 0
            return foutput
        return __hook

    prioritize = []
    for inputs, targets in tqdm(testloader, desc='Analyze'):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        mutation = torch.zeros((2, targets.size(0)), dtype=torch.long, device=device)
        confuse_classes = cfsion_mat[predicted.flatten()].topk(5, dim=1).indices
        for c in confuse_classes.unique():
            for lname, chns in susp[f'class{c.item()}'].items():
                module = rgetattr(model, lname)
                for chn in chns:
                    handle = module.register_forward_hook(_mask_out_channel(chn))
                    happen = torch.any(confuse_classes == c.item(), dim=1)
                    mutated = model(inputs).max(1).indices.eq(c.item())
                    kill = torch.logical_and(mutated, happen).long()
                    mutation[0] += kill
                    mutation[1] += happen
                    handle.remove()
        prioritize.append((mutation[0] / mutation[1]).cpu())

    prioritize = torch.cat(prioritize)
    result = sorted(
        [(i, x) for i, x in enumerate(prioritize.tolist())],
        key=lambda z: z[1], reverse=True
    )
    return result


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    valloader = load_dataloader(opt, split='val')
    testloader = load_dataloader(opt, split='test')

    with open(get_output_location(opt, 'susp_filters.json')) as f:
        susp_filters = json.load(f)
    cfsion_mat = evaluate(opt, model, valloader, device)
    priority = mutation_analyze(model, testloader, device, susp_filters, cfsion_mat)
    export_object(opt, 'mutation_kill_priority.json', priority)


if __name__ == "__main__":
    main()

