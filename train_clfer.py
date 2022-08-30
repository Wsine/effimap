import torch
from tqdm import tqdm

from dataset import load_dataloader
from model import load_model
from arguments import parser
from utils import *


def train(model, trainloader, optimizer, criterion, device, desc='Train'):
    model.train()
    train_loss, correct, total = 0, 0, 0
    with tqdm(trainloader, desc=desc) as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            avg_loss = train_loss / (batch_idx + 1)
            acc = 100. * correct / total
            tepoch.set_postfix(loss=avg_loss, acc=acc)

    acc = 100. * correct / total
    return acc


@torch.no_grad()
def validate(model, valloader, criterion, device):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with tqdm(valloader, desc='Eval') as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            avg_loss = test_loss / (batch_idx + 1)
            acc = 100. * correct / total
            tepoch.set_postfix(loss=avg_loss, acc=acc)

    acc = 100. * correct / total
    return acc


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)

    device = get_device(ctx)
    model = load_model(ctx).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    start_epoch = -1
    best_acc = 0
    if ctx.resume:
        ckp = load_torch_object(ctx, 'model_pretrained.pt')
        assert(ckp is not None)
        model.load_state_dict(ckp['net'])
        optimizer.load_state_dict(ckp['optim'])
        scheduler.load_state_dict(ckp['sched'])
        start_epoch = ckp['epoch']
        best_acc = ckp['acc']

    if ctx.eval is True:
        testloader = load_dataloader(ctx, split='val+test')
        acc = validate(model, testloader, criterion, device)
        print('test accuracy is {:.4f}%'.format(acc))
        return

    trainloader = load_dataloader(ctx, split='train')
    valloader = load_dataloader(ctx, split='val+test')
    for epoch in range(start_epoch + 1, 7):
        print('Epoch: {}'.format(epoch))
        train(model, trainloader, optimizer, criterion, device)
        acc = validate(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            save_object(ctx, state, 'model_pretrained.pt')
            best_acc = acc
            print('best accuracy is {:.4f}%'.format(acc))
        scheduler.step()


if __name__ == '__main__':
    main()

