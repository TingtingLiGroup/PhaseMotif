import pandas as pd
from torch.utils.data import dataset
import torch
import numpy as np


class Mydataset(dataset.Dataset):
    def __init__(self, path):
        super(Mydataset, self).__init__()
        self.oneHotPath = pd.read_csv(path)['onehotPath']
        self.alphabetPath = pd.read_csv(path)['alphabetPath']
        self.label = pd.read_csv(path)['Label']
        self.protein = pd.read_csv(path)['Index']

    def __getitem__(self, index):
        data_oneHot = np.loadtxt(self.oneHotPath[index])
        data_alphabet = np.loadtxt(self.alphabetPath[index])
        label = self.label[index]
        protein = self.protein[index]

        data_oneHot = torch.from_numpy(np.array(data_oneHot)).float()
        data_alphabet = torch.from_numpy(np.array(data_alphabet)).float()
        data_oneHot = data_oneHot.unsqueeze(0)  # add channel 1
        data_alphabet = data_alphabet.unsqueeze(0)
        label = torch.from_numpy(np.array(label))

        return data_oneHot, data_alphabet, label, protein

    def __len__(self):
        return len(self.label)


def my_collate(batch):  # no auto stack
    data_oneHot = [item[0] for item in batch]
    data_alphabet = [item[1] for item in batch]
    target = [item[2] for item in batch]
    protein = [item[3] for item in batch]
    return [data_oneHot, data_alphabet, target, protein]


class MyAutoencoderDataset(dataset.Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data.values, dtype=torch.float32)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
















