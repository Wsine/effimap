import torch


def get_model(_, pretrained):
    repo = 'chenyaofo/pytorch-cifar-models'
    model_name = 'cifar100_resnet32'
    model = torch.hub.load(
        repo, model_name,
        pretrained=pretrained,
        trust_repo=True
    )
    return model

