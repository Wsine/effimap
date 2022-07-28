import os
from ast import literal_eval

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


label2index = {
    'clouds': 0, 'sky': 1, 'person': 2, 'street': 3, 'window': 4, 'tattoo': 5,
    'wedding': 6, 'animal': 7, 'cat': 8, 'buildings': 9, 'tree': 10,
    'airport': 11, 'plane': 12, 'water': 13, 'grass': 14, 'cars': 15,
    'road': 16, 'snow': 17, 'sunset': 18, 'railroad': 19, 'train': 20,
    'flowers': 21, 'plants': 22, 'house': 23, 'military': 24, 'horses': 25,
    'nighttime': 26, 'lake': 27, 'rocks': 28, 'waterfall': 29, 'sun': 30,
    'vehicle': 31, 'sports': 32, 'reflection': 33, 'temple': 34, 'statue': 35,
    'ocean': 36, 'town': 37, 'beach': 38, 'tower': 39, 'toy': 40,
    'book': 41, 'bridge': 42, 'fire': 43, 'mountain': 44, 'rainbow': 45,
    'garden': 46, 'police': 47, 'coral': 48, 'fox': 49, 'sign': 50,
    'dog': 51, 'cityscape': 52, 'sand': 53, 'dancing': 54, 'leaf': 55,
    'tiger': 56, 'moon': 57, 'birds': 58, 'food': 59, 'cow': 60,
    'valley': 61, 'fish': 62, 'harbor': 63, 'bear': 64, 'castle': 65,
    'boats': 66, 'running': 67, 'glacier': 68, 'swimmers': 69, 'elk': 70,
    'frost': 71, 'protest': 72, 'soccer': 73, 'flags': 74, 'zebra': 75,
    'surf': 76, 'whales': 77, 'computer': 78, 'earthquake': 79, 'map': 80
}


class NUSWide(Dataset):
    def __init__(self,
            root_dir, mode='train',
            transform=None, target_transform=None):
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.base_dir = os.path.join(root_dir, 'nus-wide')

        csv_file = os.path.join(self.base_dir, 'nus_wid_data.csv')
        self.data = self.parse_from_csv(csv_file)

    def parse_from_csv(self, csv_file):
        df = pd.read_csv(csv_file, converters={"label": literal_eval})
        df = df[df['split_name'] == self.mode]  # type: ignore
        return df

    def __len__(self):
        return len(self.data)  # type: ignore

    def __getitem__(self, idx):
        item = self.data.iloc[idx]  # type: ignore

        img_file = os.path.join(self.base_dir, item['filepath'])
        with open(img_file, 'rb') as f:
            img = Image.open(f).convert('RGB')
        indexes = list(map(lambda x: label2index[x], item['label']))
        targets = np.zeros((self.get_num_classes(),), dtype=np.int32)
        targets[indexes] = 1

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            targets = self.target_transform(targets)

        return img, targets

    def get_num_classes(self):
        return len(label2index)


def get_dataset(opt, split, **kwargs):
    train = True if split == 'train' else False
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trsf = [
        T.RandomResizedCrop(448, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std)
    ] if train is True else [
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]
    dataset = NUSWide(
        opt.data_dir, mode='train' if train is True else 'val',
        transform=T.Compose(trsf), **kwargs
    )
    return dataset


if __name__ == '__main__':
    import torch
    dataset = NUSWide('data', mode='train')
    print(len(dataset))
    for img, tags in dataset:
        print(img)
        print(tags)
        print(torch.from_numpy(tags))
        break
