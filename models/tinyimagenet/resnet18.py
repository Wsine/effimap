from torch import nn
import torchvision
from torchvision.models.resnet import ResNet18_Weights

from utils import load_torch_object

def get_model(ctx, pretrained):
    # Load ResNet18
    model_ft = torchvision.models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained is False else None
    )
    # Finetune Final few layers to adjust for tiny imagenet input
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 200)
    w224_state = load_torch_object(ctx, 'resnet18_224_w.pt')
    model_ft.load_state_dict(w224_state['net'])
    model_ft.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    model_ft.maxpool = nn.Sequential()  # type: ignore

    if pretrained is True:
        state = load_torch_object(ctx, 'model_pretrained.pt')
        model_ft.load_state_dict(state['net'])
        print('model weights loaded.')

    return model_ft

