import torch
from torch.autograd import Variable
import torch.utils.data
from tqdm import tqdm

from dataset import load_dataloader
from model import load_model
from arguments import parser
from metric import correctness, post_predict
from utils import *


@torch.no_grad()
def evaluate(ctx, model, valloader, device):
    model.eval()

    correct_indicators = []
    for inputs, targets in tqdm(valloader, desc='Eval'):
        inputs, targets = inputs.to(device), targets.to(device)
        input_preds = post_predict(ctx, model(inputs))
        correct = correctness(ctx, input_preds, targets)
        correct_indicators.append(correct)

    correct_indicators = torch.cat(correct_indicators)
    return correct_indicators


def train(model, trainloader, optimizer, device):
    model.train()
    train_loss, rec_err = 0, 0
    for batch_idx, (inputs, targets) in enumerate(
            tepoch := tqdm(trainloader, desc='Train')):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        # random_noise = torch.where(
        #     torch.bernoulli(torch.ones(inputs.size()) * 0.1).bool(),
        #     torch.normal(0, 0.1, inputs.size()),
        #     torch.zeros(inputs.size())
        # ).to(device)
        # outputs[1] = outputs[1] + random_noise

        loss, rec_loss, _ = model.loss_function(inputs, *outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        rec_err += rec_loss.item()

        avg_loss = train_loss / (batch_idx + 1)
        avg_rec_err = rec_err / (batch_idx + 1)
        tepoch.set_postfix(loss=avg_loss, rec_err=avg_rec_err)


@torch.no_grad()
def validate(model, corloader, incloader, device):
    model.eval()

    loaders = {
        'Correct': corloader,
        'Incorrect': incloader
    }

    cor_rec_loss, inc_rec_loss = None, None
    for desc, loader in loaders.items():
        test_loss, rec_err = 0, 0
        num_batch = 0
        for batch_idx, (inputs, targets) in enumerate(
                tepoch := tqdm(loader, desc=desc)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss, rec_loss, _ = model.loss_function(inputs, *outputs)
            test_loss += loss.item()
            rec_err += rec_loss.item()

            avg_loss = test_loss / (batch_idx + 1)
            avg_rec_err = rec_err / (batch_idx + 1)
            tepoch.set_postfix(loss=avg_loss, rec_err=avg_rec_err)
            num_batch += 1
        if desc == 'Correct':
            cor_rec_loss = rec_err / num_batch
        else:
            inc_rec_loss = rec_err / num_batch
    assert(cor_rec_loss is not None and inc_rec_loss is not None)

    energy = inc_rec_loss - cor_rec_loss
    return energy


def main():
    ctx = parser.parse_args()
    print(ctx)
    guard_folder(ctx)

    device = get_device(ctx)
    model = load_model(ctx, pretrained=True).to(device)
    vae = load_model(ctx, 'vanilla_vae', pretrained=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    trainloader = load_dataloader(ctx, split='train')
    valloader = load_dataloader(ctx, split='val+test')

    correct_indicators = evaluate(ctx, model, valloader, device)
    incorrect_indice = correct_indicators.ne(1).nonzero().flatten().tolist()
    incloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(valloader.dataset, incorrect_indice),
        batch_size=ctx.batch_size, shuffle=False, num_workers=8
    )
    correct_indice = correct_indicators.nonzero().flatten().tolist()
    corloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(valloader.dataset, correct_indice),
        batch_size=ctx.batch_size, shuffle=False, num_workers=8
    )


    best_energy, num_epochs = 0, 10
    for epoch in range(num_epochs):
        print('Epoch: {}'.format(epoch))
        train(vae, trainloader, optimizer, device)
        energy = validate(vae, corloader, incloader, device)
        if epoch > num_epochs // 2 and energy > best_energy:
            best_energy = energy
            print('Saving with energy {:.6f} ...'.format(energy))
            state = {
                'epoch': epoch,
                'net': vae.state_dict(),
                'optim': optimizer.state_dict(),
                # 'sched': scheduler.state_dict(),
                'energy': energy
            }
            save_object(ctx, state, 'vae_model.pt')
        # scheduler.step()


if __name__ == '__main__':
    main()

