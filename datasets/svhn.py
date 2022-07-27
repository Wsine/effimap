import torchvision
import torchvision.transforms as T


def get_dataset(opt, split, **kwargs):
    mean = std = (0.5, 0.5, 0.5)
    trsf = [T.ToTensor(), T.Normalize(mean, std)]
    dataset = torchvision.datasets.SVHN(
        root=opt.data_dir,
        split='train' if split == 'train' else 'test',
        transform=T.Compose(trsf),
        target_transform=lambda t: int(t) - 1,
        **kwargs
    )
    return dataset


