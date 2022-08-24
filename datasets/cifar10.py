import torchvision
import torchvision.transforms as T


def get_dataset(opt, split, **kwargs):
    train = True if split == 'train' else False
    # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151?permalink_comment_id=2851662#gistcomment-2851662
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    trsf = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip()
    ] if train is True else []
    trsf += [T.ToTensor(), T.Normalize(mean, std)]
    dataset = torchvision.datasets.CIFAR10(
        root=opt.data_dir, train=train,
        transform=T.Compose(trsf),
        **kwargs
    )
    return dataset

