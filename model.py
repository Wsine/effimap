import torch


def load_model(opt):
    if 'cifar' in opt.dataset:
        model_hub = 'chenyaofo/pytorch-cifar-models'
        model_name = f'{opt.dataset}_{opt.model}'
        model = torch.hub.load(model_hub, model_name, pretrained=True)
        return model
    else:
        raise ValueError('Invalid dataset name')


def get_device(opt):
    if opt.device == 'cpu':
        return torch.device('cpu')
    elif opt.device == 'cuda':
        if torch.cuda.is_available() is False:
            return torch.device('cpu')
        return torch.device(f'cuda:{opt.gpu}')
    else:
        raise ValueError('Invalid device type')

