import torchvision
import torchvision.transforms as T


def get_dataset(opt, split, **kwargs):
    mean = std = (0.5, 0.5, 0.5)
    trsf = ([
        T.Pad(4),
        T.RandomCrop(96),
        T.RandomHorizontalFlip()
    ] if split == 'train' is True else []) \
        + [T.ToTensor(), T.Normalize(mean, std)]  # type: ignore
    dataset = torchvision.datasets.STL10(
        root=opt.data_dir,
        split='train' if split == 'train' else 'test',
        transform=T.Compose(trsf),
        **kwargs
    )
    return dataset


