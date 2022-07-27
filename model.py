import os

import torch
import torchvision

from models.fc_wrapper import FcWrapper
from utils import get_output_location, load_object, rsetattr


def load_model(opt):
    if opt.dataset in ('cifar10', 'cifar100'):
        model_hub = 'chenyaofo/pytorch-cifar-models'
        model_name = f'{opt.dataset}_{opt.model}'
        model = torch.hub.load(model_hub, model_name, pretrained=True)
    elif 'tinyimagenet' in opt.dataset:
        # refer from: https://github.com/tjmoon0104/Tiny-ImageNet-Classifier
        model = getattr(torchvision.models, opt.model)(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, opt.num_classes)
        model.conv1 = torch.nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        model.maxpool = torch.nn.Sequential()
        if os.path.exists(get_output_location(opt, 'finetune_model.pt')):
            state = load_object(opt, 'finetune_model.pt')
            restrict = True
            if state['net']['fc.weight'].size() != model.fc.weight.size():  # type: ignore
                state['net'].pop('fc.weight', None)  # type: ignore
                state['net'].pop('fc.bias', None)  # type: ignore
                restrict = False
            model.load_state_dict(state['net'], strict=restrict)  # type: ignore
            print('Finetune weights loaded.')
    elif opt.dataset == 'nuswide':
        pretrained = not os.path.exists(get_output_location(opt, 'multilabels_model.pt'))
        model = FcWrapper(opt.model, opt.num_classes, pretrained)
        if not pretrained:
            state = load_object(opt, 'multilabels_model.pt')
            model.load_state_dict(state['net'])  # type: ignore
            print('Multi-labels model weights loaded.')
    else:
        model = torch.hub.load('models', opt.model, source='local', pretrained=True)

    if opt.task == 'regress':
        last_fc = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)][-1]
        name, module = last_fc
        rsetattr(model, name, torch.nn.Linear(module.in_features, out_features=1, bias=True))
        if os.path.exists(get_output_location(opt, 'regressor_model.pt')):
            state = load_object(opt, 'regressor_model.pt')
            model.load_state_dict(state['net'])  # type: ignore
            print('Regressor weights loaded.')

    return model


def get_device(opt):
    if opt.device == 'cpu':
        return torch.device('cpu')
    elif opt.device == 'cuda':
        if torch.cuda.is_available() is False:
            return torch.device('cpu')
        return torch.device(f'cuda:{opt.gpu}')
    else:
        raise ValueError('Invalid device type')

