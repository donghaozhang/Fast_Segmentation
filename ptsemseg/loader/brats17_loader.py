import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import cv2
import nibabel
import SimpleITK as sitk

from torch.utils import data
DEBUG = True


def log(s):
    if DEBUG:
        print(s)


def load_3d_volume_as_array(filename):
    if('.nii' in filename):
        return load_nifty_volume_as_array(filename)
    elif('.mha' in filename):
        return load_mha_volume_as_array(filename)
    raise ValueError('{0:} unspported file format'.format(filename))


def load_mha_volume_as_array(filename):
    img = sitk.ReadImage(filename)
    nda = sitk.GetArrayFromImage(img)
    return nda


def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data


class Brats17Loader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=None):
        self.root = root
        self.split = split
        self.img_size = [256, 256]
        self.is_transform = is_transform
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 2
        self.files = collections.defaultdict(list)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        train_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/train_names_66.txt'
        text_file = open(train_names_path, "r")
        lines = text_file.readlines()
        img_name = self.files[index]
        img_num = np.random.randint(0, 66)
        log('The current image number is {}'.format(img_num))
        cur_im_name = lines[img_num]
        cur_im_name = cur_im_name.replace("\n", "")
        print('I am so confused', os.path.basename(cur_im_name))
        # print('the name after splitting is ', cur_im_name.split("|\")[0])
        img_path = self.root + '/' + cur_im_name + '/' + os.path.basename(cur_im_name)

        # T1 img
        t1_img_path = img_path + '_t1.nii.gz'
        t1_img = load_nifty_volume_as_array(filename=t1_img_path, with_header=False)
        log(t1_img_path)
        log('The shape of t1 img is {}'.format(t1_img.shape))

        # T1ce img
        t1ce_img_path = img_path + '_t1ce.nii.gz'
        t1ce_img = load_nifty_volume_as_array(filename=t1ce_img_path, with_header=False)
        log(t1ce_img_path)
        log('The shape of t1ce img is {}'.format(t1ce_img.shape))

        # Flair img
        flair_img_path = img_path + '_flair.nii.gz'
        flair_img = load_nifty_volume_as_array(filename=flair_img_path, with_header=False)
        log(flair_img_path)
        log('The shape of flair img is {}'.format(flair_img.shape))

        # T2 img
        t2_img_path = img_path + '_t2.nii.gz'
        t2_img = load_nifty_volume_as_array(filename=flair_img_path, with_header=False)
        log(t2_img_path)
        log('The shape of t1ce img is {}'.format(t2_img.shape))

        # segmentation path
        lbl_path = img_path + '_seg.nii.gz'
        log(lbl_path)

        img = np.asarray(img)
        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        lbl = self.encode_segmap(lbl)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
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
    dst = Brats17Loader(local_path, is_transform=True)
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