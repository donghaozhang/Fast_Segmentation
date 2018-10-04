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
from random import randint

brtas17_path = '/home/donghao/Desktop/donghao/isbi2019/code/brats17/config17/test_names_36.txt'
text_file = open(train_names_path, "r")
lines = text_file.readlines()
img_name = self.files[index]
img_num = np.random.randint(0, 66)
log('The current image number is {}'.format(img_num))
cur_im_name = lines[img_num]
cur_im_name = cur_im_name.replace("\n", "")
# print('I am so confused', os.path.basename(cur_im_name))
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

# segmentation label
lbl_path = img_path + '_seg.nii.gz'
lbl = load_nifty_volume_as_array(filename=lbl_path, with_header=False)
log(lbl_path)
log('The shape of label map img is {}'.format(t2_img.shape))

img = np.stack((t1_img, t2_img, t1ce_img, flair_img))
