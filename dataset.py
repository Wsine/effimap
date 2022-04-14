import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from datasets.tinyimagenet import TinyImageNetDataset

def load_dataset(opt, split, single_class=None, filter_idx=None, download=True):
    train = True if split == 'train' else False

    if opt.dataset == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        trsf = [T.ToTensor(), T.Normalize(mean, std)]
        dataset = torchvision.datasets.MNIST(
            root=opt.data_dir, train=train, download=download,
            transform=T.Compose(trsf)
        )
    elif opt.dataset == 'svhn':
        mean = std = (0.5, 0.5, 0.5)
        trsf = [T.ToTensor(), T.Normalize(mean, std)]
        dataset = torchvision.datasets.SVHN(
            root=opt.data_dir, download=download,
            split='train' if split == 'train' else 'test',
            transform=T.Compose(trsf)
        )
    elif opt.dataset == 'stl10':
        mean = std = (0.5, 0.5, 0.5)
        trsf = ([
            T.Pad(4),
            T.RandomCrop(96),
            T.RandomHorizontalFlip()
        ] if train is True else []) \
            + [T.ToTensor(), T.Normalize(mean, std)]  # type: ignore
        dataset = torchvision.datasets.STL10(
            root=opt.data_dir, download=download,
            split='train' if split == 'train' else 'test',
            transform=T.Compose(trsf)
        )
    elif opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        trsf = ([T.RandomCrop(32, padding=4)] if train is True else []) \
            + [T.ToTensor(), T.Normalize(mean, std)]  # type: ignore
        dataset = torchvision.datasets.CIFAR10(
            root=opt.data_dir, train=train, download=download,
            transform=T.Compose(trsf)
        )
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        trsf = ([T.RandomCrop(32, padding=4)] if train is True else []) \
            + [T.ToTensor(), T.Normalize(mean, std)]  # type: ignore
        dataset = torchvision.datasets.CIFAR100(
            root=opt.data_dir, train=train, download=download,
            transform=T.Compose(trsf)
        )
    elif opt.dataset == 'tinyimagenet':
        trsf = ([T.RandomHorizontalFlip()] if train is True else []) \
            + [T.Resize(224, T.InterpolationMode.BICUBIC), T.ToTensor()]  # type: ignore
            #  + [T.ToTensor()]  # type: ignore
        dataset = TinyImageNetDataset(
            root_dir=opt.data_dir, download=download,
            # there are no labels in test split
            mode='train' if train is True else 'val',
            transform=T.Compose(trsf)
        )
    else:
        raise ValueError('Invalid dataset name')

    labels_key = 'labels' if opt.dataset == 'stl10' else 'targets'
    if split == 'train':
        pass
    elif split == 'val':
        _, dataset = train_test_split(
            dataset, test_size=1./10, random_state=opt.seed, stratify=getattr(dataset, labels_key))
    elif split == 'test':
        dataset, _ = train_test_split(
            dataset, test_size=1./10, random_state=opt.seed, stratify=getattr(dataset, labels_key))
    else:
        raise ValueError('Invalid split parameter')

    if single_class is not None:
        clx_indices = [
            i for i, (_, y) in enumerate(dataset)  # type: ignore
            if y == single_class
        ]
        dataset = torch.utils.data.Subset(dataset, clx_indices)  # type: ignore
    if filter_idx is not None:
        dataset = torch.utils.data.Subset(dataset, filter_idx)  # type: ignore

    return dataset


def load_dataloader(opt, split, sampler=None, **kwargs):
    dataset = load_dataset(opt, split, **kwargs)
    shuffle = True if split == 'train' else False
    dataloader = torch.utils.data.DataLoader(
        dataset,  # type: ignore
        sampler=sampler,
        batch_size=opt.batch_size, shuffle=shuffle, num_workers=32
    )
    return dataloader

