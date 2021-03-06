import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

class ModifiedMNISTDataset(Dataset):
    def __init__(self, root_path, transform=None, test=False):
        self.root_path = root_path
        self.transform = transform
        self.test = test
        if not self.test:
            self.y = pd.read_csv(os.path.join(self.root_path,
                                              'train_max_y.csv'))
            self.x = pd.read_pickle(os.path.join(self.root_path,
                                              'train_max_x'))
        else:
            self.x = pd.read_pickle(os.path.join(self.root_path,
                                              'test_max_x'))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.x[idx]

        if self.transform:
            sample = self.transform(image)

        if not self.test:
            hnum = self.y['Label'][idx]
            sample = {'image': image, 'hnum': hnum}
        else:
            sample = {'image': image}

        return sample

    def show_image(self, image, idx, highest_num):
        plt.figure(figsize=(4,3))
        plt.imshow(image, label="", cmap='gray', vmin=0, vmax=255)
        plt.gca().text(5, 5, F'idx: {idx}, num: {highest_num}',
                       bbox={'facecolor': 'white', 'pad': 10})
        plt.colorbar()
