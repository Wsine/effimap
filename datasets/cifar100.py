import torchvision
import torchvision.transforms as T


def get_dataset(opt, split, **kwargs):
    train = True if split == 'train' else False
    # https://github.com/chenyaofo/image-classification-codebase/blob/master/conf/cifar100.conf
    mean = (0.5070, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2761)
    trsf = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip()
    ] if train is True else []
    trsf += [T.ToTensor(), T.Normalize(mean, std)]
    dataset = torchvision.datasets.CIFAR100(
        root=opt.data_dir, train=train,
        transform=T.Compose(trsf),
        **kwargs
    )
    return dataset

