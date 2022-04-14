dependencies = ['torch']
from playground.mnist.model import mnist as mnist_loader
from playground.svhn.model import svhn as svhn_loader
from playground.stl10.model import stl10 as stl10_loader


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
