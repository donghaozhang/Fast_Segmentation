import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import cv2

from torch.utils import data


class brats17Loader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=None):
        self.root = root
        self.split = split
        self.img_size = [256, 256]
        self.is_transform = is_transform
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 2
        self.files = collections.defaultdict(list)

        # for split in ["train", "test", "val"]:
        #     file_list = os.listdir(root + '/' + split)
        #     self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # img_name = self.files[self.split][index]
        train_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/train_names_66.txt'
        text_file = open(train_names_path, "r")
        lines = text_file.readlines()
        print(lines[0])
        img_name = self.files[index]
        print(img_name)
        print('the value of index is ', index)
        # img_path = self.root + '/' + self.split + '/' + img_name
        img_num = np.random.randint(0, 67)
        img_path = self.root + '/' + lines[img_num]

        print('the current img_path is ', img_path)
        print(len(lines))
        lbl_path = self.root + '/' + self.split + 'annot/' + img_name
        print(lbl_path)
        img = cv2.imread(img_path)
        img = np.asarray(img)
        img = np.array(img, dtype=np.uint8)
        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)
        if self.is_transform:
            img, lbl = self.transform(img, lbl)
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


if __name__ == '__main__':
    # local_path = '/home/neuron/Desktop/Donghao/cellsegmentation/normalCV/cell_cancer_dataset'
    # local_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/maskrcnn/compare'
	# local_path = '/home/donghao/Desktop/donghao/brain_segmentation/brain_data_full/HGG'
    dst = brats17Loader(local_path, is_transform=True)
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