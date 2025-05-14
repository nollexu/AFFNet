"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from skimage.util import view_as_windows
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import SimpleITK as sitk


class Traditional_Dataset(data_utils.Dataset):
    def __init__(self, data_list, label_list, is_training=True):
        self.data_list, self.label_list = data_list, label_list
        self.is_training = is_training

    def __len__(self):
        return len(self.data_list)

    def load(self, image_path):
        # print(image_path)
        vol = sitk.Image(sitk.ReadImage(image_path))
        inputsize = vol.GetSize()
        # print(f'inputsize{inputsize}')
        inputspacing = vol.GetSpacing()
        # print(f'inputspacing{inputspacing}')
        if self.is_training == True:
            rand1 = torch.rand(1)[0]
            if rand1 >= 0 and rand1 < 0.166:
                # print('方案1')
                img_array = sitk.GetArrayFromImage(vol)[0:160, 12:204, 10:170]
            elif rand1 >= 0.166 and rand1 < 0.333:
                img_array = sitk.GetArrayFromImage(vol)[2:162, 12:204, 10:170]
                # print('方案2')
            elif rand1 >= 0.333 and rand1 < 0.499:
                img_array = sitk.GetArrayFromImage(vol)[1:161, 11:203, 10:170]
                # print('方案3')
            elif rand1 >= 0.499 and rand1 < 0.666:
                img_array = sitk.GetArrayFromImage(vol)[1:161, 13:205, 10:170]
                # print('方案4')
            elif rand1 >= 0.666 and rand1 < 0.833:
                img_array = sitk.GetArrayFromImage(vol)[1:161, 12:204, 9:169]
                # print('方案5')
            else:
                img_array = sitk.GetArrayFromImage(vol)[1:161, 12:204, 11:171]
                # print('方案6')
        else:
            # print('测试集')
            img_array = sitk.GetArrayFromImage(vol)[1:161, 12:204, 10:170]
        # 查看图像像素的最大值以及最小值
        # print(f'image value range: [{img_array.min()}, {img_array.max()}]')
        # 忘记归一化，试试归一化后。
        # print(img_array.max())
        # 以一定的概率左右脑反转一下
        if self.is_training == True:
            rand = torch.rand(1)[0]
            # print('训练集')
            if rand > 0.5:
                # print('翻转一下')
                img_array = img_array[:, :, ::-1]
        img_array = img_array / img_array.max()
        return torch.Tensor(img_array).unsqueeze(dim=0)

    def __getitem__(self, index):
        # print('index', index)
        path = self.data_list[index]
        # print('path', path)
        label = self.label_list[index]
        # print('label', label)
        tensor = self.load(path)
        # print('tensor', tensor.shape)
        return tensor, int(label)
