import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import gaussian_kde
# from sklearn.neighbors import KernelDensity

from dataset import load_dataloader
from model import load_model
from arguments import parser
from utils import *


activation_trace = []
def activation_hook(module, finputs, foutputs):
    if foutputs.dim() == 4:
        # For convolutional layers
        # https://github.com/coinse/sadl/blob/master/sa.py#L94
        at = foutputs.flatten(start_dim=2).mean(dim=2)
    else:
        at = foutputs.flatten(start_dim=1)
    at = at.detach().cpu().numpy()
    activation_trace.append(at)


@torch.no_grad()
def train_lsa_kde_function(ctx, model, trainloader, device):
    if ctx.dataset == 'cifar100' and ctx.model == 'resnet32':
        model.layer1[4].bn1.register_forward_hook(activation_hook)
        num_classes = 100
    else:
        raise NotImplemented

    class_ats = {c: [] for c in range(num_classes)}
    for inputs, targets in tqdm(trainloader, desc='Trace'):
        activation_trace.clear()
        model(inputs.to(device))
        ats = activation_trace[0]
        for at, t in zip(ats, targets):
            class_ats[t.item()].append(at)

    temp = {}
    for c in class_ats.keys():
        temp[c] = np.stack(class_ats[c], axis=0)
    class_ats = temp

    cap_trace_len = class_ats[0].shape[1]
    print('captured tract length:', cap_trace_len)
    remove_cols = set()
    for ats in class_ats.values():
        for col in range(cap_trace_len):
            if np.var(ats[:, col]) < 1e-5:
                remove_cols.add(col)
    keep_cols = [col for col in range(cap_trace_len) if col not in remove_cols]
    reduced_trace_len = len(keep_cols)
    print('reduced trace length:', reduced_trace_len)

    # scott method for computing bandwidth
    # from: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neighbors/_kde.py#L217
    # bd = train_at.shape[0] ** (-1 / (train_at.shape[1] + 4))
    # kde = KernelDensity(kernel='gaussian', bandwidth=bd)
    # kde.fit(train_at)

    kdes = {}
    for c in range(num_classes):
        kdes[c] = gaussian_kde(class_ats[c][:, keep_cols])

    output = { 'kde': kdes, 'keep_cols': keep_cols }
    save_object(ctx, output, 'lsa_kde_functions.pkl')


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)

    device = get_device(ctx)
    model = load_model(ctx, pretrained=True).to(device).eval()
    trainloader = load_dataloader(ctx, split='train')

    train_lsa_kde_function(ctx, model, trainloader, device)


if __name__ == '__main__':
    main()

