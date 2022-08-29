import copy

import torch
from tqdm import tqdm

from dataset import load_dataloader
from model import load_model
from arguments import parser
from utils import *


def train_and_validate(ctx, model, dataloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_acc = 0
    best_state = None
    num_epochs = 100
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        for phase in ('train', 'val'):
            if phase == 'train':
                model.train()
            else:
                model.eval()

            correct, total = 0, 0
            for inputs, targets in tqdm(dataloader[phase], desc=phase):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = outputs.max(1)
                    loss = criterion(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    correct += preds.eq(targets).sum().item()
                total += targets.size(0)

            if phase == 'val':
                acc = 100. * correct / total
                if acc > best_acc:
                    print(f'Update snapshotor with acc: {acc:.4f}%...')
                    best_acc = acc
                    best_state = copy.deepcopy(model).cpu()
        scheduler.step()

    return best_state


def train_snapshotors(ctx, model, tloader, vloader, device):
    if ctx.dataset == 'cifar100' and ctx.model == 'resnet32':
        snapshotors = {
            'num_classes': 100,
            'layers': ['relu', 'layer1', 'layer2', 'layer3'],
            'in_features': [16, 16, 32, 64]
        }
    elif ctx.dataset == 'tinyimagenet' and ctx.model == 'resnet18':
        snapshotors = {
            'num_classes': 200,
            'layers': ['relu', 'layer1', 'layer2', 'layer3', 'layer4'],
        }
    else:
        raise ValueError('Not supported combinations now')

    dataloader = {'train': tloader, 'val': vloader}

    module_seq = torch.nn.Sequential()
    for name, layer in model.named_children():
        module_seq.append(layer)
        if name in snapshotors['layers']:
            print('Processing submodel up to {} layer'.format(name))
            layer_idx = snapshotors['layers'].index(name)
            in_features = snapshotors['in_features'][layer_idx]
            out_features = snapshotors['num_classes']
            submodel = copy.deepcopy(module_seq)
            submodel.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            submodel.append(torch.nn.Flatten())
            submodel.append(torch.nn.Linear(in_features, out_features))
            submodel = submodel.to(device)

            model_state = train_and_validate(ctx, submodel, dataloader, device)
            save_object(ctx, model_state, f'snapshotors/snapshotor_{name}.pt')


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx, folder='snapshotors')

    device = get_device(ctx)
    model = load_model(ctx)
    trainloader = load_dataloader(ctx, split='train')
    valloader = load_dataloader(ctx, split='val+test')

    train_snapshotors(ctx, model, trainloader, valloader, device)


if __name__ == '__main__':
    main()

