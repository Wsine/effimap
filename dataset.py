import os

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from datasets.tinyimagenet import TinyImageNetDataset
from datasets.noisycifar import NCIFAR10, NCIFAR100


def load_dataset(opt, split, download=True):
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
            transform=T.Compose(trsf),
            target_transform=lambda t: int(t) - 1
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
    elif opt.dataset in ('cifar10', 'ncifar10'):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        trsf = ([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] \
                if train is True else []) \
             + [T.ToTensor(), T.Normalize(mean, std)]  # type: ignore
        if opt.dataset == 'ncifar10':
            dataset = NCIFAR10(
                root=opt.data_dir, transform=T.Compose(trsf),
                noise_type='random_label2',  # 'random_label1'/2/3 'worse_label'
                noise_path=os.path.join(opt.data_dir, 'noisycifar/data/CIFAR-10_human.pt')
            )
        else:
            dataset = torchvision.datasets.CIFAR10(
                root=opt.data_dir, train=train, download=download,
                transform=T.Compose(trsf)
            )
    elif opt.dataset in ('cifar100', 'ncifar100'):
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        trsf = ([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] \
                if train is True else []) \
             + [T.ToTensor(), T.Normalize(mean, std)]  # type: ignore
        if opt.dataset == 'ncifar100':
            dataset = NCIFAR100(
                root=opt.data_dir, transform=T.Compose(trsf),
                noise_type='noisy_label',
                noise_path=os.path.join(opt.data_dir, 'noisycifar/data/CIFAR-100_human.pt')
            )
        else:
            dataset = torchvision.datasets.CIFAR100(
                root=opt.data_dir, train=train, download=download,
                transform=T.Compose(trsf)
            )
    elif 'tinyimagenet' in opt.dataset:
        trsf = ([T.RandomHorizontalFlip()] if train is True else []) \
            + [T.ToTensor()]  # type: ignore
            #  + [T.Resize(224, T.InterpolationMode.BICUBIC), T.ToTensor()]  # type: ignore
        # trg_trsf = (lambda y: y - 100) if 'trf' in opt.dataset else None
        trg_trsf = None
        dataset = TinyImageNetDataset(
            root_dir=opt.data_dir, download=download,
            # there are no labels in test split
            mode='train' if train is True else 'val',
            transform=T.Compose(trsf),
            target_transform=trg_trsf
        )
    else:
        raise ValueError('Invalid dataset name')

    if opt.dataset in ('stl10', 'svhn'):
        labels_key = 'labels'
    elif opt.dataset.startswith('ncifar'):
        labels_key = 'train_noisy_labels'
    else:
        labels_key = 'targets'
    if split == 'train':
        if opt.dataset.startswith('ncifar'):
            dataset, _ = train_test_split(
                dataset, test_size=1./5, random_state=opt.seed, stratify=getattr(dataset, labels_key))
    elif split == 'val':
        if opt.dataset.startswith('ncifar'):
            _, dataset = train_test_split(
                dataset, test_size=1./50, random_state=opt.seed, stratify=getattr(dataset, labels_key))
        else:
            _, dataset = train_test_split(
                dataset, test_size=1./10, random_state=opt.seed, stratify=getattr(dataset, labels_key))
    elif split == 'test':
        if opt.dataset.startswith('ncifar'):
            _, dataset = train_test_split(
                dataset, test_size=9./50, random_state=opt.seed, stratify=getattr(dataset, labels_key))
        else:
            dataset, _ = train_test_split(
                dataset, test_size=1./10, random_state=opt.seed, stratify=getattr(dataset, labels_key))
    else:
        raise ValueError('Invalid split parameter')

    if opt.dataset == 'tinyimagenet-trf':
        clx_indices = [i for i, (_, y) in enumerate(dataset) if y >= 0 and y < 100]  # type: ignore
        dataset = torch.utils.data.Subset(dataset, clx_indices)  # type: ignore

    return dataset


def load_dataloader(opt, split, sampler=None, shuffle=None, **kwargs):
    dataset = load_dataset(opt, split, **kwargs)
    if shuffle is None:
        shuffle = True if split == 'train' else False
    dataloader = torch.utils.data.DataLoader(
        dataset,  # type: ignore
        sampler=sampler,
        batch_size=opt.batch_size, shuffle=shuffle, num_workers=32
    )
    return dataloader


if __name__ == '__main__':
    from arguments import parser
    opt = parser.parse_args()
    print(opt)
    dataset = load_dataset(opt, 'val')
