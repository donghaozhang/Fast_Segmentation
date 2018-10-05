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

from torch.utils import data

DEBUG = False


def log(s):
	if DEBUG:
		print(s)


def load_3d_volume_as_array(filename):
	if ('.nii' in filename):
		return load_nifty_volume_as_array(filename)
	elif ('.mha' in filename):
		return load_mha_volume_as_array(filename)
	raise ValueError('{0:} unspported file format'.format(filename))


def load_mha_volume_as_array(filename):
	img = sitk.ReadImage(filename)
	nda = sitk.GetArrayFromImage(img)
	return nda


def load_nifty_volume_as_array(filename, with_header=False):
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
	data = np.transpose(data, [2, 1, 0])
	if (with_header):
		return data, img.affine, img.header
	else:
		return data


def convert_label(in_volume, label_convert_source, label_convert_target):
	"""
	convert the label value in a volume
	inputs:
		in_volume: input nd volume with label set label_convert_source
		label_convert_source: a list of integers denoting input labels, e.g., [0, 1, 2, 4]
		label_convert_target: a list of integers denoting output labels, e.g.,[0, 1, 2, 3]
	outputs:
		out_volume: the output nd volume with label set label_convert_target
	"""
	mask_volume = np.zeros_like(in_volume)
	convert_volume = np.zeros_like(in_volume)
	for i in range(len(label_convert_source)):
		source_lab = label_convert_source[i]
		target_lab = label_convert_target[i]
		if (source_lab != target_lab):
			temp_source = np.asarray(in_volume == source_lab)
			temp_target = target_lab * temp_source
			mask_volume = mask_volume + temp_source
			convert_volume = convert_volume + temp_target
	out_volume = in_volume * 1
	out_volume[mask_volume > 0] = convert_volume[mask_volume > 0]
	return out_volume


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
		return 66

	def __getitem__(self, index):
		train_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/train_names_66.txt'
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
		patch_size = [80, 120, 120]
		log('the patch_size is {} {} {}'.format(patch_size[0], patch_size[1], patch_size[2]))
		x_start = randint(0, img.shape[1] - patch_size[0])
		x_end = x_start + patch_size[0]
		y_start = randint(0, img.shape[2] - patch_size[1])
		y_end = y_start + patch_size[1]
		z_start = randint(0, img.shape[3] - patch_size[2])
		z_end = z_start + patch_size[2]
		log('x_start is {} x_end is {}'.format(x_start, x_end))
		log('The shape of image after stacking is : {}'.format(img.shape))
		img = img[:, x_start:x_end, y_start:y_end, z_start:z_end]
		log('The shape of image after stacking is : {}'.format(img.shape))
		img = np.asarray(img)
		img = np.array(img, dtype=np.uint8)
		lbl = np.array(lbl, dtype=np.int32)
		lbl = convert_label(lbl, [0, 1, 2, 4], [0, 1, 2, 3])
		log('The shape of label is : {}'.format(lbl.shape))
		lbl = lbl[x_start:x_end, y_start:y_end, z_start:z_end]
		# img (4, 155, 240, 240)
		# label (155, 240, 240)

		log('The maximum value of img is {}'.format(np.max(img)))
		log('The unique values of label {}'.format(np.unique(lbl)))
		log('!!!!!!! I should convert labels of [0 1 2 4] into [0, 1, 2, 3]!!!!!')
		# transform is disabled for now
		if self.is_transform:
			img, lbl = self.transform(img, lbl)

		# convert numpy type into torch type
		img = torch.from_numpy(img).float()
		lbl = torch.from_numpy(lbl).long()
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
		return np.asarray(
			[[128, 128, 128], [128, 0, 0], [192, 192, 128], [255, 69, 0], [128, 64, 128], [60, 40, 222], [128, 128, 0],
			 [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]])

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
	dst = Brats17Loader()
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
