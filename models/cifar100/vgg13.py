import torch


def get_model(_, pretrained):
    repo = 'chenyaofo/pytorch-cifar-models'
    model_name = 'cifar100_vgg13_bn'
    model = torch.hub.load(
        repo, model_name,
        pretrained=pretrained,
        trust_repo=True
    )
    return model

