from os import listdir
from os.path import sep

from PIL import Image
import numpy as np
from numpy.random import multivariate_normal
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class LocalDataset(Dataset):
    def __init__(self, data_src: str, num_classes=None, train=True, train_prop=0.8, transform=None):
        super().__init__()

        self.data_src = data_src
        self.class_names = sorted(listdir(data_src))
        self.transform = transform

        self.num_classes = num_classes
        if not num_classes:
            self.num_classes = len(self.class_names)

        self.id2name = {i: name for i, name in enumerate(self.class_names)}
        self.name2id = {name: i for i, name in enumerate(self.class_names)}
        self.im_paths = {self.name2id[name]: [sep.join([data_src, name, im_name]) for im_name in listdir(sep.join([data_src, name]))] for name in self.class_names}

        self.all_samples = []
        for class_id in self.im_paths.keys():
            append_list = [(class_id, im_path) for im_path in self.im_paths[class_id]]
            cutoff_index = int(len(append_list) * train_prop)
            if train:
                append_list = append_list[:cutoff_index]
            else:
                append_list = append_list[cutoff_index:]
            self.all_samples += append_list
        np.random.shuffle(self.all_samples)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, item):
        data_pair = self.all_samples[item]
        #image = imageio.v2.imread(data_pair[1])
        image = Image.open(data_pair[1]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, data_pair[0]


class InAndOutLocalDataset(Dataset):
    def __init__(self, data_src: str, data_ref_src: str, train=True, in_distribution=True, num_classes=None, train_prop=0.8, transform=None, max_samples=None):
        super().__init__()
        np.random.seed(0)

        self.data_src = data_src
        self.data_ref_src = data_ref_src
        self.transform = transform

        # load class reference information
        ref_data = pd.read_csv(data_ref_src)
        grouped = ref_data.groupby('data_class')['species_name'].apply(list).to_dict()
        self.class_names = sorted(grouped['ID']) if in_distribution else sorted(grouped['OOD'])
        self.num_classes = num_classes
        if not num_classes:
            self.num_classes = len(self.class_names)

        self.id2name = {i: name for i, name in enumerate(self.class_names)}
        self.name2id = {name: i for i, name in enumerate(self.class_names)}
        self.im_paths = {
            self.name2id[name]: [sep.join([data_src, name, im_name]) for im_name in listdir(sep.join([data_src, name]))]
            for name in self.class_names}
        if max_samples:
            for key in self.im_paths.keys():
                np.random.shuffle(self.im_paths[key])
                self.im_paths[key] = self.im_paths[key][:min(len(self.im_paths[key]), max_samples)]

        self.all_samples = []
        for class_id in self.im_paths.keys():
            append_list = [(class_id, im_path) for im_path in self.im_paths[class_id]]
            cutoff_index = int(len(append_list) * train_prop)
            if train:
                append_list = append_list[:cutoff_index]
            else:
                append_list = append_list[cutoff_index:]
            self.all_samples += append_list
        np.random.shuffle(self.all_samples)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, item):
        data_pair = self.all_samples[item]
        # image = imageio.v2.imread(data_pair[1])
        image = Image.open(data_pair[1]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, data_pair[0]

