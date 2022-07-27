import torchvision
import torchvision.transforms as T


def get_dataset(opt, split, **kwargs):
    train = True if split == 'train' else False
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    trsf = ([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] \
            if train is True else []) \
         + [T.ToTensor(), T.Normalize(mean, std)]  # type: ignore
    dataset = torchvision.datasets.CIFAR10(
        root=opt.data_dir, train=train,
        transform=T.Compose(trsf),
        **kwargs
    )
    return dataset


