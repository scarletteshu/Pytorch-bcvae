from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import transforms
import numpy as np
import torch
import math
import configs as cfg


class DataSet(Dataset):
    def __init__(self, tensor):
        self.imgs = tensor

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, item):
        return self.imgs[item]


class Dataloader():

    def __init__(self):
        dataset_zip = np.load(cfg.params['data_path'], allow_pickle=True, encoding="latin1")

        self.all_imgs = torch.from_numpy(dataset_zip['imgs']).unsqueeze(1).float()

        self.train_data = DataSet(self.all_imgs[:491520])
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.params['batch_size'],
                                       shuffle=True,
                                       num_workers=2,
                                       pin_memory=True,
                                       drop_last=True)

        self.val_data = DataSet(self.all_imgs[491520:])
        self.val_loader = DataLoader(self.val_data,
                                     batch_size=cfg.params['batch_size'],
                                     shuffle=True,
                                     num_workers=2,
                                     pin_memory=True,
                                     drop_last=True)
