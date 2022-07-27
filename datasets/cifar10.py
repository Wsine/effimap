import torchvision
import torchvision.transforms as T


def get_dataset(opt, split, **kwargs):
    train = True if split == 'train' else False
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    trsf = ([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] \
            if train is True else []) \
         + [T.ToTensor(), T.Normalize(mean, std)]  # type: ignore
    dataset = torchvision.datasets.CIFAR10(
        root=opt.data_dir, train=train,
        transform=T.Compose(trsf),
        **kwargs
    )
    return dataset


