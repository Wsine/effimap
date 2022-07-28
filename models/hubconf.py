import torch

from vendor.playground.mnist.model import mnist as mnist_loader
from vendor.playground.svhn.model import svhn as svhn_loader
from vendor.playground.stl10.model import stl10 as stl10_loader
from vendor.MSRN.models import MSGDN


def mlp(pretrained=False, **kwargs):
    model = mnist_loader(pretrained=pretrained, **kwargs)
    return model


def svhn(pretrained=False, **kwargs):
    model = svhn_loader(
        n_channel=32, pretrained=pretrained, **kwargs
    )
    return model


def stl10(pretrained=False, **kwargs):
    model = stl10_loader(
        n_channel=32, pretrained=pretrained, **kwargs
    )
    return model


def msgdn(pretrained=False, **kwargs):
    backbone = 'resnet101'
    graph_file = 'vendor/MSRN/data/nuswide/nuswide_adj.pkl'
    pool_ratio = 0.05
    num_classes = 81
    model = MSGDN(
        num_classes, pool_ratio, backbone, graph_file, **kwargs
    )
    if pretrained is True:
        ckpt = torch.load('data/nus-wide/MSGDN-nus-wide.pth.tar')
        model.load_state_dict(ckpt['state_dict'], strict=False)
    return model
