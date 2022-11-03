import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_hook(chn, sample_idx=None):
    def _hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            if sample_idx is not None:
                output[sample_idx, chn] = 0
            else:
                output[:, chn] = 0
        elif isinstance(module, nn.BatchNorm2d):
            if sample_idx is not None:
                output[sample_idx, chn] = inputs[0][sample_idx, chn]
            else:
                output[:, chn] = inputs[0][:, chn]
        elif isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            try:
                d = output.get_device()
                d = torch.device(f'cuda:{d}')
            except RuntimeError:
                d = torch.device('cpu')
            g = torch.Generator(d).manual_seed(chn)
            mask = torch.bernoulli(
                torch.where(output[0] < 0, 0.1, 0.0), generator=g
            ).bool()
            if sample_idx is not None:
                output[sample_idx] = torch.where(
                    mask, inputs[0][sample_idx], output[sample_idx]
                )
            else:
                output = torch.where(mask.expand(output.size()), inputs[0], output)
        return output
    return _hook


def get_tensor_mask(tensor, seed, ratio=0.1):
    g = torch.Generator().manual_seed(seed)
    num_neurons = tensor.numel()
    num_mask = int(num_neurons * ratio)
    shuffle = torch.randperm(num_neurons, generator=g)
    mask = torch.cat(
        (torch.ones((num_mask,), dtype=torch.long),
         torch.zeros((num_neurons - num_mask,), dtype=torch.long))
    )[shuffle].reshape_as(tensor).bool()
    return mask


def NAI_hook(rand_seed):
    def _hook(module, inputs):
        mask = get_tensor_mask(inputs[0][0], rand_seed)
        try:
            device = inputs[0].get_device()
            mask = mask.cuda(device)
        except RuntimeError:
            pass
        for batch_input in inputs:  # tuple
            for single_input in batch_input:
                single_input = torch.where(mask, single_input.mul(-1.), single_input)
        return inputs
    return _hook


def NEB_hack(tensor, rand_seed, fill=0, **kwargs):
    mask = get_tensor_mask(tensor, rand_seed, **kwargs)
    tensor.masked_fill_(mask, fill)
    return tensor


def GF_hack(tensor, rand_seed, std=0.1, **kwargs):
    mask = get_tensor_mask(tensor, rand_seed, **kwargs)
    g = torch.Generator().manual_seed(rand_seed)
    gauss = torch.normal(0, std, tensor.size(), generator=g)
    if isinstance(tensor, nn.parameter.Parameter):
        tensor.copy_(torch.where(mask, tensor + gauss, tensor))
    else:
        tensor = torch.where(mask, tensor + gauss, tensor)
    return tensor


def WS_hack(tensor, rand_seed, ratio=0.1):
    mask = get_tensor_mask(tensor, rand_seed, ratio)
    mask_indices = mask.nonzero(as_tuple=True)
    g = torch.Generator().manual_seed(rand_seed)
    shuffle = torch.randperm(mask_indices[0].size(0), generator=g)
    tensor[mask_indices] = tensor[mask_indices][shuffle]
    return tensor


def PCR_hack(tensor, rand_seed, **kwargs):
    mask = get_tensor_mask(tensor, rand_seed, **kwargs)
    tensor = torch.where(mask, 1 - tensor, tensor)
    return tensor


feature_container = []
def feature_hook(module, inputs, outputs):
    dims = [i for i in range(1, outputs.dim())]
    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU)):
        mean = torch.mean(outputs, dim=dims)
        feature_container.append(mean)
        var = torch.var(outputs, dim=dims)
        feature_container.append(var)
    if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
        in_mask = (inputs[0] < 0).sum(dim=dims) / outputs[0].numel()
        out_mask = (outputs < 0).sum(dim=dims) / outputs[0].numel()
        feature_container.append(out_mask - in_mask)
    elif isinstance(module, nn.Linear):
        mean = torch.mean(inputs[0], dim=dims)
        feature_container.append(mean)
        var = torch.var(inputs[0], dim=dims)
        feature_container.append(var)
        probs = F.softmax(outputs, dim=1)
        # gini = F.softmax(outputs, dim=1).square().sum(dim=1).mul(-1.).add(1.)
        # feature_container.append(gini)
        # entropy = torch.log(F.softmax(outputs, dim=1).max(dim=1).values).mul(-1.)
        entropy = probs.mul(torch.log(probs)).sum(dim=1).mul(-1.)  # shannon entropy
        feature_container.append(entropy)

