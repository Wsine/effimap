import os

import torch
import torchvision

from utils import get_output_location, load_object


def load_model(opt):
    if 'cifar' in opt.dataset:
        model_hub = 'chenyaofo/pytorch-cifar-models'
        model_name = f'{opt.dataset}_{opt.model}'
        model = torch.hub.load(model_hub, model_name, pretrained=True)
    elif opt.dataset == 'tinyimagenet':
        # refer from: https://github.com/tjmoon0104/Tiny-ImageNet-Classifier
        model = getattr(torchvision.models, opt.model)(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 200)
        if os.path.exists(get_output_location(opt, 'finetune_model.pt')):
            state = load_object(opt, 'finetune_model.pt')
            model.load_state_dict(state['net'])  # type: ignore
            print('Finetune weights loaded.')
    else:
        model = torch.hub.load('models', opt.model, source='local', pretrained=True)
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


if __name__ == '__main__':
    opt = type('',(object,),{'model': 'resnet18', 'dataset': 'tinyimagenet'})()
    model = load_model(opt)
    print(model)
