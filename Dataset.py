import os
import sys
import time
import numpy as np
import pandas as pd

# Read and write TIFF images
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# PyTorch for ML
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, RandomRotation, ToPILImage


class ImageDataset(Dataset):

    def __init__(self, df, split, im_size=224):
        self.df = df
        self.split = split
        self.all_image_names = self.df[:]['id']
        self.all_labels = np.array(self.df.drop(['id', 'Names'], axis=1))[:,1:]
        train_p = 0.75
        num_train = int(train_p * len(self.df))
        num_test = 10
        num_valid = len(self.df) - num_train - num_test

        if split == 'train':

            self.transform = Compose([
                Resize((im_size, im_size)),
                RandomHorizontalFlip(p=0.5),
                RandomRotation(degrees=45),
                ToTensor(),
            ])

            self.image_names = list(self.all_image_names[:num_train])
            self.labels = list(self.all_labels[:num_train])

        elif split == 'valid':
            self.image_names = list(self.all_image_names[-(num_valid + num_test):-num_test])
            self.labels = list(self.all_labels[-(num_valid + num_test):-num_test])
            self.transform = Compose([
                Resize((im_size, im_size)),
                ToTensor(),
            ])

        elif split == 'test':
            self.image_names = list(self.all_image_names[-num_test:])
            self.labels = list(self.all_labels[-num_test:])
            self.transform = Compose([
                Resize((im_size, im_size)),
                ToTensor(),
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        im = Image.open(self.image_names[index])
        im = self.transform(im)
        targets = self.labels[index]
        image = torch.tensor(im, dtype=torch.float32)
        if image.shape[0] < 3:
            image=torch.tile(image,[3,1,1])
        return {
            'image': image,
            'label': torch.tensor(targets, dtype=torch.float32)
        }
