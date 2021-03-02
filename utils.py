import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def checkOS(file_name):
    if os.name == 'nt':
        return file_name.replace("\\", "/")
    else:
        return file_name

class MNIST(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transforms
        self.frame = pd.read_csv(checkOS(self.root_dir + "\\" + csv_file))

        # train data shape => (42000, 785) -> 785(real data=1~785)
        # convert >> (42000, 28, 28)
        self.label = self.frame.label.values
        self.data = np.reshape(self.frame.values[:,1:], (self.frame.shape[0], 28, 28))


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.frame.iloc[idx, 1:].astype(np.uint8).values.reshape(28, 28)
        label = self.frame.iloc[idx, 0].astype(np.uint8).values
        if self.transform:
            img = self.transform(img)
        return label, img

def prepare_dataloaders(args):
    data_path = os.path.join(os.getcwd(), 'Datas')

    train_transform = torchvision.transforms.Compose([
                               transforms.RandomCrop(32, padding=4), #padding=4
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)),
                           ])
    valid_transform = torchvision.transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)),
                           ])

    train_dataset = MNIST(csv_file = 'train.csv',
                        root_dir=data_path,
                        transform=train_transform)

    valid_dataset = MNIST(csv_file = 'test.csv',
                        root_dir=data_path,
                        transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size = args.batch,
                                                num_workers = args.workers)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size = args.batch,
                                                num_workers = args.workers)


    return train_loader, valid_loader, len(train_dataset), len(valid_dataset)

def prepare_test_dataloaders(args):
    data_path = os.path.join(os.getcwd(), 'Datas')
    test_transform = torchvision.transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)),
                           ])

    test_dataset = MNIST(csv_file = 'test.csv',
                        root_dir=data_path,
                        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size = 1,
                                                num_workers = args.workers)


    return test_loader, len(test_dataset)
