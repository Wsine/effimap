import importlib

import torch
import torch.utils.data
from sklearn.model_selection import train_test_split


def load_dataset(opt, split, **kwargs):
    data_module = importlib.import_module(f'datasets.{opt.dataset}')
    dataset = data_module.get_dataset(opt, split, **kwargs)

    stratify = dataset.targets if hasattr(dataset, 'targets') else \
               dataset.labels  if hasattr(dataset, 'labels') else \
               None
    if split == 'train' or split == 'val+test':
        pass
    elif split == 'val':
        _, val_indices = train_test_split(
            list(range(len(dataset))),
            test_size=1./10, random_state=opt.seed, stratify=stratify
        )
        dataset = torch.utils.data.Subset(dataset, val_indices)  # type: ignore
    elif split == 'test':
        test_indices, _ = train_test_split(
            list(range(len(dataset))),
            test_size=1./10, random_state=opt.seed, stratify=dataset.targets
        )
        dataset = torch.utils.data.Subset(dataset, test_indices)  # type: ignore
    else:
        raise ValueError('Invalid split parameter')

    return dataset


def load_dataloader(opt, split, shuffle=None, **kwargs):
    dataset = load_dataset(opt, split, **kwargs)
    if shuffle is None:
        shuffle = True if split == 'train' else False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size, shuffle=shuffle, num_workers=8
    )
    return dataloader


if __name__ == '__main__':
    from arguments import parser
    opt = parser.parse_args()
    print(opt)
    dataloader = load_dataloader(opt, 'train')
    batch = iter(dataloader).next()
    (imgs, inps), targets = batch
    print(imgs.size())
    print(inps.size())
    print(targets.size())
