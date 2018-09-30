import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import cv2

from torch.utils import data


class cellcancerLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=None):
        self.root = root
        self.split = split
        self.img_size = [256, 256]
        # self.img_size = [512, 512]
        self.is_transform = is_transform
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 2
        self.files = collections.defaultdict(list)

        for split in ["train", "test", "val"]:
            file_list = os.listdir(root + '/' + split)
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + '/' + self.split + '/' + img_name
        lbl_path = self.root + '/' + self.split + 'annot/' + img_name
        # print(img_path)
        # print('index value ', index)
        # if img_path == '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val1/comp_exp/train/555.png':
        #     print('bingo')
        img = cv2.imread(img_path)
        img = np.asarray(img)
        # img = img.astype(float)
        # img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        # print(img.shape)
        lbl = m.imread(lbl_path)

        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        # print('label shape: ', lbl.shape)
        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        # print('image shape is ', img.shape)
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # print('the current size of the image is ', self.img_size[0], self.img_size[1])
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        # print('transform is being called')
        img = torch.from_numpy(img).float()

        lbl = self.encode_segmap(lbl)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        # print('the shape of lbl after resize is ', lbl.shape)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def get_camvid_labels(self):
        return np.asarray([[128, 128, 128], [128, 0, 0], [192, 192, 128], [255, 69, 0], [128, 64, 128], [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]])

    def get_cellcancer_labels(self):
        return np.asarray([[200, 0, 0], [0, 0, 0]])

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_cellcancer_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, temp, plot=False):
        cell_foreground = [200, 0, 0]
        cell_background = [0, 0, 0]
        label_colours = np.array([cell_foreground, cell_background])
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb
    # def decode_segmap(self, temp, plot=False):
    #     Sky = [128, 128, 128]
    #     Building = [128, 0, 0]
    #     Pole = [192, 192, 128]
    #     Road_marking = [255, 69, 0]
    #     Road = [128, 64, 128]
    #     Pavement = [60, 40, 222]
    #     Tree = [128, 128, 0]
    #     SignSymbol = [192, 128, 128]
    #     Fence = [64, 64, 128]
    #     Car = [64, 0, 128]
    #     Pedestrian = [64, 64, 0]
    #     Bicyclist = [0, 128, 192]
    #     Unlabelled = [0, 0, 0]
    #     label_colours = np.array([Sky, Building, Pole, Road_marking, Road,
    #                               Pavement, Tree, SignSymbol, Fence, Car,
    #                               Pedestrian, Bicyclist, Unlabelled])
    #     r = temp.copy()
    #     g = temp.copy()
    #     b = temp.copy()
    #     for l in range(0, self.n_classes):
    #         r[temp == l] = label_colours[l, 0]
    #         g[temp == l] = label_colours[l, 1]
    #         b[temp == l] = label_colours[l, 2]
    #
    #     rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    #     rgb[:, :, 0] = r
    #     rgb[:, :, 1] = g
    #     rgb[:, :, 2] = b
    #     if plot:
    #         plt.imshow(rgb)
    #         plt.show()
    #     else:
    #         return rgb

if __name__ == '__main__':
    # local_path = '/home/neuron/Desktop/Donghao/cellsegmentation/normalCV/cell_cancer_dataset'
    local_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/maskrcnn/compare'
    dst = cellcancerLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            plt.imshow(dst.decode_segmap(labels.numpy()[i]))
            plt.show()