import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class NUSWide(Dataset):
    def __init__(self,
            root_dir, mode='train',
            transform=None, target_transform=None):
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.base_dir = os.path.join(root_dir, 'nus-wide')
        self.img_dir = os.path.join(self.base_dir, 'images')

        csv_file = os.path.join(self.base_dir, f'{mode}_data.csv')
        if os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file)
        else:
            self.data = self.parse2csv(mode, csv_file)

    def parse2csv(self, mode, csv_file):
        ERROR_STRING = 'http://farm2.static.flickr.com/1331/566491861_3d5b3'
        urls_file = os.path.join(self.base_dir, 'NUS-WIDE-urls.txt')
        df = pd.read_table(
            urls_file, delim_whitespace=True,
            header=None, skiprows=1,
            names=[
                'Photo_file', 'Photo_id', 'url_Large',
                'url_Middle', 'url_Small', 'url_Original'
            ],
            usecols=['Photo_file', 'url_Small'],
            engine='python',
            # bad line happens on line 67
            on_bad_lines=lambda x: [y.replace(ERROR_STRING, '') for y in x[:2]]
        )
        assert(isinstance(df, pd.DataFrame))
        df['Photo_file'] = df['Photo_file'].map(lambda x: x[20:])
        df['url_Small'] = df['url_Small'].map(
            lambda x: os.path.basename(x) if isinstance(x, str) else x)

        labels_dir = os.path.join(self.base_dir, 'AllLabels')
        for _, _, filenames in os.walk(labels_dir):
            for filename in sorted(filenames):
                label_name = filename[7:][:-4]  # extract Labels_{label_name}.txt
                label_file = os.path.join(labels_dir, filename)
                with open(label_file) as f:
                    labels = list(map(lambda x: int(x.strip()), f.readlines()))
                    df[label_name] = labels

        img_list_dir = os.path.join(self.base_dir, 'ImageList')
        list_file_name = 'TrainImagelist.txt' if mode == 'train' else 'TestImagelist.txt'
        list_file = os.path.join(img_list_dir, list_file_name)
        with open(list_file) as f:
            photo_files = list(map(lambda x: x.strip(), f.readlines()))
        # photo_files = photo_files[:10]  # for debugging only
        df = df[df['Photo_file'].isin(photo_files)]

        df = df.drop(df[df['url_Small'].isna()].index)
        m2indexed = {}
        for _, _, filenames in os.walk(self.img_dir):
            for filename in filenames:
                m_file = '_'.join(filename.split('_')[1:])
                m2indexed[m_file] = filename
        df['url_Small'] = df['url_Small'].apply(lambda x: m2indexed.get(x))
        df = df.drop(df[df['url_Small'].isna()].index)

        df.to_csv(csv_file, index=False)

        return df

    def __len__(self):
        return len(self.data)  # type: ignore

    def __getitem__(self, idx):
        item = self.data.iloc[idx]  # type: ignore

        img_file = os.path.join(self.img_dir, item['url_Small'])
        with open(img_file, 'rb') as f:
            img = Image.open(f).convert('RGB')
        tags = item.iloc[2:].to_numpy(dtype=np.float32)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            tags = self.target_transform(tags)

        return img, tags


def get_dataset(opt, split, **kwargs):
    train = True if split == 'train' else False
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trsf = [
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std)
    ] if train is True else [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]
    dataset = NUSWide(
        opt.data_dir, mode='train' if train is True else 'test',
        transform=T.Compose(trsf), **kwargs
    )
    return dataset


if __name__ == '__main__':
    import torch
    dataset = NUSWide('data', mode='test')
    print(len(dataset))
    for img, tags in dataset:
        print(img)
        print(tags)
        print(torch.from_numpy(tags))
        break
