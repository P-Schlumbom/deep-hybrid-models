import numpy as np
from numpy.random import multivariate_normal
import torch
from torch.utils.data import Dataset, DataLoader

from datasets.data_classes import BaseDataClass

class CustomDataset(Dataset):
    def __init__(self, datamodel: BaseDataClass, num_samples=5000):
        self.datamodel = datamodel
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return self.datamodel.sample()

