from torch import nn
from torchvision import models


class FcWrapper(nn.Module):
    def __init__(self, name, out_dim, pretrained=False):
        super(FcWrapper, self).__init__()

        model_factory = getattr(models, name)
        self.backbone = model_factory(pretrained=pretrained)
        self.fc = nn.Linear(1000, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

