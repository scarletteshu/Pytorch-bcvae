from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import transforms
import numpy as np
import torch
import math
import configs as cfg


class DataSet(Dataset):
    def __init__(self):
        dataset_zip = np.load(cfg.params['data_path'], allow_pickle=True, encoding="latin1")

        self.imgs = dataset_zip['imgs'].astype(dtype='float32')
        self.imgs = torch.from_numpy(self.imgs).unsqueeze(1)
        # self.latents_values = dataset_zip['latents_values']
        # self.latents_classes = dataset_zip['latents_classes']
        # metadata = dataset_zip['metadata'][()]
        # self.latents_sizes = metadata['latents_sizes']

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, item):
        return self.imgs[item]


class Dataloader():

    def __init__(self):
        self.num_train_imgs = 0
        self.num_val_imgs = 0
        self.K_cross = 10
        self.dataset = DataSet()

    def datasplit(self):
        length = math.ceil(self.dataset.__len__()/self.K_cross)
        lengths = [length for i in range(9)]
        lengths.append(self.dataset.__len__() - 9*length)
        self.datasets = random_split(self.dataset, lengths, torch.Generator().manual_seed(179719721))

    # dataloader
    def train_dataloader(self, k:int):
        train_dataset = ConcatDataset(self.datasets[:k] + self.datasets[k+1:])
        self.num_train_imgs = train_dataset.__len__()
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.params['batch_size'],
                                  shuffle=False,
                                  drop_last=True)
        return train_loader

    def val_dataloader(self, k:int):
        self.num_val_imgs = self.datasets[k].__len__()
        val_loader = DataLoader(self.datasets[k],
                                batch_size=cfg.params['batch_size'],
                                shuffle=False,
                                drop_last=True)
        return val_loader

    """
    def data_transforms(self):

            SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
            SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

            if self.params['dataset'] == 'celeba':
                transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(148),
                                                transforms.Resize(self.params['img_size']),
                                                transforms.ToTensor(),
                                                SetRange])
            else:
                raise ValueError('Undefined dataset type')
            return transform
    """

"""
if __name__ == "__main__":

   dl = Dataloader()
   #print(dl.dataset.imgs.shape)
   dl.datasplit()

   k=10
   tr = dl.train_dataloader(k-1)
   val = dl.val_dataloader(k-1)
"""