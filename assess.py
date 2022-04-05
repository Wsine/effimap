import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from dataset import load_dataloader
from model import get_device, load_model
from models.vanilla_vae import VanillaVAE
from arguments import parser
from utils import *


dispatcher = AttrDispatcher('assess')

@torch.no_grad()
def evaluate_accuracy(opt, model, dataloader, device):
    model.eval()

    correct, total = 0, 0
    confusion_matrix = torch.zeros(opt.num_classes, opt.num_classes)
    for inputs, targets in tqdm(dataloader, desc='Evaluate', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        for t, p in zip(targets, predicted):
            confusion_matrix[t.item(), p.item()] += 1

    acc = correct / total
    class_acc = (confusion_matrix.diag() / confusion_matrix.sum(1)).tolist()
    return acc, class_acc


@dispatcher.register('perfdiff')
def performance_difference(opt, model, device):
    valloader = load_dataloader(opt, split='val')
    base_acc, base_c_acc = evaluate_accuracy(opt, model, valloader, device)
    print('base acc =', base_acc)
    print('base class acc =', base_c_acc)

    def _mask_out_channel(chn):
        def __hook(module, finput, foutput):
            foutput[:, chn] = 0
            return foutput
        return __hook

    df = pd.DataFrame(columns=['layer', 'filter_idx', 'acc'] + \
                      [f'acc_c{c}' for c in range(opt.num_classes)])
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for lname in tqdm(conv_names, desc='Modules', leave=True):
        module = rgetattr(model, lname)
        for chn in tqdm(range(module.out_channels), desc='Filters', leave=False):
            handle = module.register_forward_hook(_mask_out_channel(chn))
            acc, c_acc = evaluate_accuracy(opt, model, valloader, device)
            r1 = { 'layer': lname, 'filter_idx': chn, 'acc': acc - base_acc }
            r2 = { f'acc_c{c}': c_acc[c] - base_c_acc[c] for c in range(opt.num_classes) }
            df = df.append({**r1, **r2}, ignore_index=True)
            handle.remove()

    return df, 'filter_assess.csv'


@dispatcher.register('bnrunning')
@torch.no_grad()
def bn_running_mean_std(opt, model, device):
    def _bn_hook(module, inputs, outputs):
        return F.batch_norm(
            inputs[0],
            module.freeze_running_mean, module.freeze_running_var,
            module.weight, module.bias,
            False,  # bn_training
            0.0 if module.momentum is None else module.momentum,
            module.eps
        )

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            module.freeze_running_mean = module.running_mean.clone()  # type: ignore
            module.freeze_running_var = module.running_var.clone()  # type: ignore
            module.register_forward_hook(_bn_hook)  # type: ignore

    result = {}
    for clx_idx in range(opt.num_classes):
        print('Processing class {}'.format(clx_idx))
        trainloader = load_dataloader(opt, split='train', single_class=clx_idx)

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()

        for _ in tqdm(range(opt.epochs), desc='Epoch', leave=True):
            for inputs, targets in tqdm(trainloader, desc='BN', leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                _ = model(inputs)

        result[f'c{clx_idx}'] = {
            name: {
                'running_mean': module.running_mean.cpu(),
                'running_var': module.running_var.cpu()
            }
            for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)
        }

    return result, 'bn_running_stats.pt'


@dispatcher.register('vae-train')
def execution_trace_for_vae(opt, model, device):
    model.eval()

    trainloader = load_dataloader(opt, split='train')

    equals = []
    temploader = load_dataloader(opt, split='val')
    for inputs, targets in temploader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.max(1)
        equals.append(predicted.eq(targets))
    filter_idx = torch.cat(equals).nonzero().flatten().cpu().tolist()
    valloader = load_dataloader(opt, split='val', filter_idx=filter_idx, download=False)

    vae_container = []

    #  def _bn_hook(module, inputs, outputs):
    #      bn_var1 = torch.var(inputs[0], dim=(1, 2, 3))
    #      intermediate =  F.batch_norm(
    #          inputs[0],
    #          module.running_mean, module.running_var,
    #          None, None,
    #          False,  # bn_training
    #          0.0 if module.momentum is None else module.momentum,
    #          module.eps
    #      )
    #      bn_var2 = torch.var(intermediate, dim=(1, 2, 3))
    #      bn_var3 = torch.var(outputs, dim=(1, 2, 3))
    #      vae_container.append(bn_var2 - bn_var1)
    #      vae_container.append(bn_var3 - bn_var1)
    #
    #  def _act_hook(module, inputs, outputs):
    #      mask = (inputs[0] < 0).sum(dim=(1, 2, 3)) / outputs[0].numel()
    #      vae_container.append(mask)
    #
    #  def _ln_hook(module, inputs, outputs):
    #      gini = F.softmax(outputs, dim=1).square().sum(dim=1).mul(-1.).add(1.)
    #      vae_container.append(gini)
    #
    #  last_linear = None
    #  for module in model.modules():
    #      if isinstance(module, nn.BatchNorm2d):
    #          module.register_forward_hook(_bn_hook)
    #      elif isinstance(module, nn.ReLU):
    #          module.register_forward_hook(_act_hook)
    #      elif isinstance(module, nn.Linear):
    #          last_linear = module
    #  if last_linear is not None:
    #      last_linear.register_forward_hook(_ln_hook)

    def _hook(module, inputs, outputs):
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
            mean = torch.mean(outputs, dim=(2, 3))
            var = torch.var(outputs, dim=(2, 3))
            vae_container.append(mean)
            vae_container.append(var)
        if isinstance(module, nn.ReLU):
            in_mask = (inputs[0] < 0).sum(dim=(2, 3)) / outputs[0, 0].numel()
            out_mask = (outputs < 0).sum(dim=(2, 3)) / outputs[0, 0].numel()
            vae_container.append(out_mask - in_mask)
        elif isinstance(module, nn.Linear):
            in_feat_mean = torch.mean(inputs[0], dim=1, keepdim=True)
            out_feat_mean = torch.mean(inputs[0], dim=1, keepdim=True)
            vae_container.append(out_feat_mean - in_feat_mean)
            in_feat_var = torch.var(inputs[0], dim=1, keepdim=True)
            out_feat_var = torch.var(inputs[0], dim=1, keepdim=True)
            vae_container.append(out_feat_var - in_feat_var)
            gini = F.softmax(outputs, dim=1).square().sum(dim=1, keepdim=True).mul(-1.).add(1.)
            vae_container.append(gini)

    for module in model.modules():
        module.register_forward_hook(_hook)

    vae_mean, vae_std, vae_normalize = None, 1, []
    vae_norm_file = os.path.join(
        opt.output_dir, opt.dataset, opt.model, 'vae_normalize.pt')
    if os.path.exists(vae_norm_file):
        with open(vae_norm_file, 'rb') as f:
            stat = torch.load(f, map_location=torch.device('cpu'))
            vae_mean = stat['vae_mean'].to(device)
            vae_std = stat['vae_std'].to(device)

    padded_len, padded_op = 2, None
    vae_model, optimizer, scheduler = None, None, None
    best_reconstr_err = 1e5

    for e in range(opt.epochs):
        print('Epoch: {}'.format(e))

        if vae_model is not None: vae_model.train()
        train_loss, reconstr_err = 0, 0
        tepoch = tqdm(trainloader, desc='Train')
        for batch_idx, (inputs, _) in enumerate(tepoch):
            inputs = inputs.to(device)
            vae_container.clear()

            # Extract features
            with torch.no_grad():
                model(inputs)
            #  vae_inputs = torch.stack(vae_container).transpose(0, 1)
            vae_inputs = torch.cat(vae_container, dim=1)

            # Normalize features
            if vae_mean is None:
                vae_normalize.append(vae_inputs)
                continue
            vae_inputs.sub_(vae_mean).div_(vae_std + 1e-5)  # normalize

            # VAE model preparation
            if padded_op is None:
                #  vae_len = vae_inputs.size()[-1]
                #  padded_len = vae_len
                #  while padded_len % (2 ** 5) != 0: padded_len += 1
                #  padded_op = nn.ConstantPad1d((0, padded_len - vae_len), 0)
                #  vae_model = VanillaVAE(1, padded_len, 128).to(device)
                _, padded_op = vae_std.topk(4096, largest=False)
                vae_model = VanillaVAE(1, 4096, 128).to(device)
                optimizer = torch.optim.Adam(
                    vae_model.parameters(),
                    lr=0.00005,
                    weight_decay=0.0
                )
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.95
                )
            #  vae_inputs = padded_op(vae_inputs)
            vae_inputs = vae_inputs[:, padded_op]
            vae_inputs = vae_inputs.unsqueeze(1)  # expand dimension

            assert(vae_model is not None and optimizer is not None)
            optimizer.zero_grad()
            vae_outputs = vae_model(vae_inputs)
            loss = vae_model.loss_function(*vae_outputs, M_N=0.00025)
            loss['loss'].backward()
            optimizer.step()

            train_loss += loss['loss'].item()
            reconstr_err += loss['Reconstruction_Loss'].item()
            avg_loss = train_loss / (batch_idx + 1)
            avg_rec_err = reconstr_err / (batch_idx + 1)
            tepoch.set_postfix(loss=avg_loss, err=avg_rec_err)

        if vae_mean is None:
            x = torch.cat(vae_normalize)
            vae_mean = x.mean(dim=0)
            vae_std = x.std(dim=0)
            vae_normalize.clear()

            with open(vae_norm_file, 'wb') as f:
                torch.save({'vae_mean': vae_mean, 'vae_std': vae_std}, f)
            print('Saved vae mean and std.')
            continue

        if vae_model is not None: vae_model.eval()
        test_loss, reconstr_err = 0, 0
        tepoch = tqdm(valloader, desc='Validate')
        for batch_idx, (inputs, _) in enumerate(tepoch):
            inputs = inputs.to(device)
            vae_container.clear()

            # Extract features
            with torch.no_grad():
                model(inputs)
            #  vae_inputs = torch.stack(vae_container).transpose(0, 1)
            vae_inputs = torch.cat(vae_container, dim=1)

            # Normalize features
            vae_inputs.sub_(vae_mean).div_(vae_std + 1e-5)  # normalize

            # VAE model preparation
            assert(padded_op is not None)
            #  vae_inputs = padded_op(vae_inputs)
            vae_inputs = vae_inputs[:, padded_op]
            vae_inputs = vae_inputs.unsqueeze(1)  # expand dimension

            assert(vae_model is not None and optimizer is not None)
            with torch.no_grad():
                vae_outputs = vae_model(vae_inputs)
                loss = vae_model.loss_function(*vae_outputs, M_N=1.0)

            test_loss += loss['loss'].item()
            reconstr_err += loss['Reconstruction_Loss'].item()
            avg_loss = test_loss / (batch_idx + 1)
            avg_rec_err = reconstr_err / (batch_idx + 1)
            tepoch.set_postfix(loss=avg_loss, err=avg_rec_err)

        if reconstr_err < best_reconstr_err:
            print('Saving...')
            state = {
                'epoch': e,
                'net': vae_model.state_dict(),  # type: ignore
                'optim': optimizer.state_dict(),  # type: ignore
                'sched': scheduler.state_dict(),  # type: ignore
                'err': reconstr_err,
                'padded': padded_len
            }
            torch.save(state, get_output_location(opt, 'vae_model.pt'))
            best_reconstr_err = reconstr_err

        assert(scheduler is not None)
        scheduler.step()

    return None, None


def main():
    opt = parser.add_dispatch(dispatcher).parse_args()
    print(opt)
    guard_folder(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)

    result, result_name = dispatcher(opt, model, device)
    if result is not None:
        export_object(opt, result_name, result)


if __name__ == '__main__':
    main()

