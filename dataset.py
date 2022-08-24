import importlib

import torch
import torch.utils.data
from sklearn.model_selection import train_test_split


def load_dataset(ctx, split, **kwargs):
    data_module = importlib.import_module(f'datasets.{ctx.dataset}')
    dataset = data_module.get_dataset(ctx, split, **kwargs)

    stratify = dataset.targets if hasattr(dataset, 'targets') else \
               dataset.labels  if hasattr(dataset, 'labels') else \
               None
    if split == 'train' or split == 'val+test':
        pass
    elif split == 'val':
        _, val_indices = train_test_split(
            list(range(len(dataset))),
            test_size=1./10, random_state=ctx.seed, stratify=stratify
        )
        dataset = torch.utils.data.Subset(dataset, val_indices)  # type: ignore
    elif split == 'test':
        test_indices, _ = train_test_split(
            list(range(len(dataset))),
            test_size=1./10, random_state=ctx.seed, stratify=stratify
        )
        dataset = torch.utils.data.Subset(dataset, test_indices)  # type: ignore
    else:
        raise ValueError('Invalid split parameter')

    return dataset


def load_dataloader(ctx, split, shuffle=None, **kwargs):
    dataset = load_dataset(ctx, split, **kwargs)
    if shuffle is None:
        shuffle = True if split == 'train' else False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ctx.batch_size, shuffle=shuffle, num_workers=8
    )
    return dataloader

