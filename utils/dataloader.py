# -*- coding: utf-8 -*-
"""
时间：2022年03月17日
"""
# encoding: utf-8
import torch
from torch.utils.data import Dataset
from PIL import Image


class DatasetGenerator(Dataset):
    def __init__(self, image_names, labels, transform):
        self.image_names = image_names
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        image = Image.open(image_name)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_names)


class DatasetGenerator_CON(Dataset):
    def __init__(self, image_names, image_names1, labels, transform=None):
        self.image_names = image_names
        self.image_names1 = image_names1
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_name1 = self.image_names1[index]
        label = self.labels[index]
        image = Image.open(image_name)
        image1 = Image.open(image_name1)
        if self.transform is not None:
            image = self.transform(image)
            image1 = self.transform(image1)
        return image, image1, label

    def __len__(self):
        return len(self.image_names)


class DatasetGenerator_KD(Dataset):
    def __init__(self, image_names_tea, image_names_stu, labels, transform, transform_val, img_list=None):
        self.labels = labels
        self.transform = transform
        self.transform_val = transform_val
        self.image_names_tea = image_names_tea
        self.image_names_stu = image_names_stu
        self.img_list = img_list

    def __getitem__(self, index):
        image_name_tea = self.image_names_tea[index]
        image_name_stu = self.image_names_stu[index]
        label = self.labels[index]
        im_list = self.img_list[index]
        image_tea = Image.open(image_name_tea)
        image_stu = Image.open(image_name_stu)
        image_tea = self.transform_val(image_tea)
        image_stu = self.transform(image_stu)
        return image_tea, image_stu, label, im_list

    def __len__(self):
        return len(self.image_names_tea)


if __name__ == "__main__":
    DS = DatasetGenerator_KD()
    print("x")
