# taken from: https://rosenfelder.ai/multi-input-neural-network-pytorch/
import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class NyresDataset(Dataset):
    """Tabular and Image dataset."""

    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, 'new-york-real-estate', 'processed_images')
        pickle_file = os.path.join(data_dir, 'new-york-real-estate', 'df.pkl')
        self.tabular = pd.read_pickle(pickle_file)

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tabular = self.tabular.iloc[idx, 0:]

        y = tabular["price"]

        image = Image.open(f"{self.image_dir}/{tabular['zpid']}.png")
        image = np.array(image)
        image = image[..., :3]

        image = transforms.functional.to_tensor(image)

        tabular = tabular[["latitude", "longitude", "beds", "baths", "area"]]
        tabular = tabular.tolist()
        tabular = torch.FloatTensor(tabular)

        return image, tabular, y


if __name__ == '__main__':
    dataset = NyresDataset('data')
    image, tabular, y = dataset[0]
    print(image.size(), tabular.size(), y)
