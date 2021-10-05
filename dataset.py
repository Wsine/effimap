import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split


def load_dataset(opt, split):
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_trsf = [T.RandomCrop(32, padding=4)]
    else:
        raise ValueError('Invalid dataset name')

    train = True if split == 'train' else False
    transformers = (
        train_trsf if train is True else []
    ) + [
        T.ToTensor(),
        T.Normalize(mean, std)
    ]

    dataset = torchvision.datasets.CIFAR10(
        root=opt.data_dir, train=train, download=True,
        transform=T.Compose(transformers)
    )

    if split == 'train':
        trainset = dataset
        return trainset
    if split == 'val':
        _, valset = train_test_split(
            dataset, test_size=1./10, random_state=2021, stratify=dataset.targets)
        return valset
    elif split == 'test':
        testset, _ = train_test_split(
            dataset, test_size=1./10, random_state=2021, stratify=dataset.targets)
        return testset
    else:
        raise ValueError('Invalid split parameter')


def load_dataloader(opt, split):
    dataset = load_dataset(opt, split)
    shuffle = True if split == 'train' else False
    dataloader = torch.utils.data.DataLoader(
        dataset,  # type: ignore
        batch_size=opt.batch_size, shuffle=shuffle, num_workers=2
    )
    return dataloader

