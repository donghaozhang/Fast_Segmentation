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

# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
import os
import nibabel
import numpy as np
import random
from scipy import ndimage
import SimpleITK as sitk

DEBUG = False


def log(s):
	if DEBUG:
		print(s)


def search_file_in_folder_list(folder_list, file_name):
	"""
	Find the full filename from a list of folders
	inputs:
		folder_list: a list of folders
		file_name:  filename
	outputs:
		full_file_name: the full filename
	"""
	file_exist = False
	for folder in folder_list:
		full_file_name = os.path.join(folder, file_name)
		if (os.path.isfile(full_file_name)):
			file_exist = True
			break
	if (file_exist == False):
		raise ValueError('{0:} is not found in {1:}'.format(file_name, folder))
	return full_file_name


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
	log(filename)
	img = nibabel.load(filename)
	data = img.get_data()
	data = np.transpose(data, [2, 1, 0])
	if (with_header):
		return data, img.affine, img.header
	else:
		return data


def save_array_as_nifty_volume(data, filename, reference_name=None):
	"""
	save a numpy array as nifty image
	inputs:
		data: a numpy array with shape [Depth, Height, Width]
		filename: the ouput file name
		reference_name: file name of the reference image of which affine and header are used
	outputs: None
	"""
	log(filename)
	img = sitk.GetImageFromArray(data)
	if (reference_name is not None):
		img_ref = sitk.ReadImage(reference_name)
		img.CopyInformation(img_ref)
	sitk.WriteImage(img, filename)


def itensity_normalize_one_volume(volume):
	"""
	normalize the itensity of an nd volume based on the mean and std of nonzeor region
	inputs:
		volume: the input nd volume
	outputs:
		out: the normalized nd volume
	"""

	pixels = volume[volume > 0]
	mean = pixels.mean()
	std = pixels.std()
	out = (volume - mean) / std
	out_random = np.random.normal(0, 1, size=volume.shape)
	out[volume == 0] = out_random[volume == 0]
	return out


def get_ND_bounding_box(label, margin):
	"""
	get the bounding box of the non-zero region of an ND volume
	"""
	# print('the value of margin is ', margin)
	input_shape = label.shape
	if (type(margin) is int):
		margin = [margin] * len(input_shape)
	log('The length of input_shape is {}'.format(margin))
	log('The length of margin is {}'.format(margin))
	# assert (len(input_shape) == len(margin))

	indxes = np.nonzero(label)
	idx_min = []
	idx_max = []
	for i in range(len(input_shape)):
		idx_min.append(indxes[i].min())
		idx_max.append(indxes[i].max())
	# print('idx_min: ', idx_min, 'idx_max: ', idx_max)

	for i in range(len(input_shape)):
		idx_min[i] = max(idx_min[i] - margin[i], 0)
		idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
	# print('idx_min', idx_min, 'idx_max', idx_max)
	return idx_min, idx_max


def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
	"""
	crop/extract a subregion form an nd image.
	"""
	dim = len(volume.shape)
	assert (dim >= 2 and dim <= 5)
	if (dim == 2):
		output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
							   range(min_idx[1], max_idx[1] + 1))]
	elif (dim == 3):
		output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
							   range(min_idx[1], max_idx[1] + 1),
							   range(min_idx[2], max_idx[2] + 1))]
	elif (dim == 4):
		output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
							   range(min_idx[1], max_idx[1] + 1),
							   range(min_idx[2], max_idx[2] + 1),
							   range(min_idx[3], max_idx[3] + 1))]
	elif (dim == 5):
		output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
							   range(min_idx[1], max_idx[1] + 1),
							   range(min_idx[2], max_idx[2] + 1),
							   range(min_idx[3], max_idx[3] + 1),
							   range(min_idx[4], max_idx[4] + 1))]
	else:
		raise ValueError("the dimension number shoud be 2 to 5")
	return output


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


def get_random_roi_sampling_center(input_shape, output_shape, sample_mode, bounding_box=None):
	"""
	get a random coordinate representing the center of a roi for sampling
	inputs:
		input_shape: the shape of sampled volume
		output_shape: the desired roi shape
		sample_mode: 'valid': the entire roi should be inside the input volume
					 'full': only the roi centre should be inside the input volume
		bounding_box: the bounding box which the roi center should be limited to
	outputs:
		center: the output center coordinate of a roi
	"""
	center = []
	for i in range(len(input_shape)):
		if (sample_mode[i] == 'full'):
			if (bounding_box):
				x0 = bounding_box[i * 2];
				x1 = bounding_box[i * 2 + 1]
			else:
				x0 = 0;
				x1 = input_shape[i]
		else:
			if (bounding_box):
				x0 = bounding_box[i * 2] + int(output_shape[i] / 2)
				x1 = bounding_box[i * 2 + 1] - int(output_shape[i] / 2)
			else:
				x0 = int(output_shape[i] / 2)
				x1 = input_shape[i] - x0
		if (x1 <= x0):
			centeri = int((x0 + x1) / 2)
		else:
			centeri = random.randint(x0, x1)
		center.append(centeri)
	return center


def extract_roi_from_volume(volume, in_center, output_shape, fill='random'):
	"""
	extract a roi from a 3d volume
	inputs:
		volume: the input 3D volume
		in_center: the center of the roi
		output_shape: the size of the roi
		fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
	outputs:
		output: the roi volume
	"""
	input_shape = volume.shape
	if (fill == 'random'):
		output = np.random.normal(0, 1, size=output_shape)
	else:
		output = np.zeros(output_shape)
	# print('the output_shape is ', output_shape)
	r0max = [int(x / 2) for x in output_shape]
	# print('r0max: ', r0max)
	r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
	# print('r1max: ', r1max)
	r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
	# print('r0: ', r0)
	r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
	# print('r1: ', r1)
	out_center = r0max

	output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
				  range(out_center[1] - r0[1], out_center[1] + r1[1]),
				  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
		volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
					  range(in_center[1] - r0[1], in_center[1] + r1[1]),
					  range(in_center[2] - r0[2], in_center[2] + r1[2]))]
	return output


class RandomFlipInsideOut3d(object):
	def __init__(self, p):
		self.p = p  # [0,1)

	def __call__(self, img, mask, rd_int):
		if random.random() < self.p:
			return (
				(np.flip(img, axis=rd_int)).copy(), (np.flip(mask, axis=rd_int)).copy()
			)
		return img, mask


class RandomRotate3d(object):
	def __init__(self, degree):
		self.degree = degree  # -180, 180

	def __call__(self, img, mask):
		from scipy.ndimage import rotate
		rotate_degree = random.random() * 2 * self.degree - self.degree  # -degree, +degree
		rotation_plane = random.sample(range(0, 3), 2)
		return (
			rotate(img, rotate_degree, rotation_plane).copy(),
			rotate(mask, rotate_degree, rotation_plane).copy()
		)


class Brats17Loader(data.Dataset):
	def __init__(self, root, split="train", is_transform=False, img_size=None):
		self.root = root
		self.split = split
		self.img_size = [256, 256]
		self.is_transform = is_transform
		self.mean = np.array([104.00699, 116.66877, 122.67892])
		self.n_classes = 2
		self.files = collections.defaultdict(list)
		self.train_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/train_names_66.txt'
		# self.train_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/brats17/config17/train_names_all.txt'
		self.text_file = open(self.train_names_path, "r")
		self.lines = self.text_file.readlines()
		log('The length lines is {}'.format(len(self.lines)))

	def __len__(self):
		return len(self.lines)

	def __getitem__(self, index):
		# train_names_path =
		# text_file = open(train_names_path, "r")

		img_name = self.files[index]
		img_num = np.random.randint(0, len(self.lines))
		log('The current image number is {}'.format(img_num))
		cur_im_name = self.lines[img_num]
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
		# print('The lbl path is {}'.format(lbl_path))
		lbl = load_nifty_volume_as_array(filename=lbl_path, with_header=False)
		lbl = np.array(lbl, dtype=np.int32)
		log(lbl_path)
		log('The shape of label map img is {}'.format(t2_img.shape))
		log('The unique values of lbl_patch {}'.format(np.unique(lbl)))
		########### Previous data loading: Begin
		img = np.stack((t1_img, t2_img, t1ce_img, flair_img))
		patch_size = [64, 64, 64]
		log('the patch_size is {} {} {}'.format(patch_size[0], patch_size[1], patch_size[2]))
		lbl_patch_length = 1
		# while lbl_patch_length == 1:
		idx_min, idx_max = get_ND_bounding_box(label=lbl, margin=[0, 0, 0, 0])
		margin = 10
		idx_min[0] = idx_min[0] - margin
		idx_min[1] = idx_min[1] - margin
		idx_min[2] = idx_min[2] - margin
		# img = img[:,idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]]
		# lbl = lbl[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]]
		# print('idx_min is {} idx_max is {}'.format(idx_min, idx_max))
		x_start = randint(idx_min[0], img.shape[1] - patch_size[0])
		# x_start = randint(8, img.shape[1] - patch_size[0] - 8)
		x_end = x_start + patch_size[0]
		# y_start = randint(0+8, img.shape[2] - patch_size[1]-8)
		y_start = randint(idx_min[1], img.shape[2] - patch_size[1])
		y_end = y_start + patch_size[1]
		z_start = randint(idx_min[2], img.shape[3] - patch_size[2])
		z_end = z_start + patch_size[2]
		lbl_patch = lbl[x_start:x_end, y_start:y_end, z_start:z_end]
		log('The unique value of lbl patch is {}'.format(np.unique(lbl_patch)))
		lbl_patch_length = 1
		log('The length value list is {}'.format(len(np.unique(lbl_patch))))
		lbl_patch = np.array(lbl_patch, dtype=np.int32)
		lbl_patch = convert_label(lbl_patch, [0, 1, 2, 4], [0, 1, 2, 3])
		log('The shape of label is : {}'.format(lbl_patch.shape))

		log('x_start is {} x_end is {}'.format(x_start, x_end))
		log('The shape of image after stacking is : {}'.format(img.shape))
		img_patch = img[:, x_start:x_end, y_start:y_end, z_start:z_end]
		log('The shape of image after stacking is : {}'.format(img.shape))
		img_patch = np.asarray(img_patch)
		flip_int = np.random.randint(0, 4)
		# print('The number of flip_int is ', flip_int)
		if flip_int < 2.5:
			t1_patch = img_patch[0, :, :, :]
			t1_patch = (np.flip(t1_patch, axis=flip_int)).copy()
			t2_patch = img_patch[1, :, :, :]
			t2_patch = (np.flip(t2_patch, axis=flip_int)).copy()
			t1ce_patch = img_patch[2, :, :, :]
			t1ce_patch = (np.flip(t1ce_patch, axis=flip_int)).copy()
			flair_patch = img_patch[3, :, :, :]
			flair_patch = (np.flip(flair_patch, axis=flip_int)).copy()
			img_patch[0, :, :, :] = t1_patch.copy()
			img_patch[1, :, :, :] = t1ce_patch.copy()
			img_patch[2, :, :, :] = t2_patch.copy()
			img_patch[3, :, :, :] = flair_patch.copy()
			# img_patch = (np.flip(img_patch, axis=flip_int)).copy()
			# print('The shape of img is {}'.format(img_patch.shape))
			lbl_patch = (np.flip(lbl_patch, axis=flip_int)).copy()
			# print('The shape of lbl is {}'.format(lbl_patch.shape))

		# img = np.array(img, dtype=np.uint8)

		##img (4, 155, 240, 240)
		##label (155, 240, 240)

		log('The maximum value of img is {}'.format(np.max(img)))
		log('The unique values of label {}'.format(np.unique(lbl)))
		log('!!!!!!! I should convert labels of [0 1 2 4] into [0, 1, 2, 3]!!!!!')
		lbl_patch = convert_label(in_volume=lbl_patch, label_convert_source=[0, 1, 2, 4],
								  label_convert_target=[0, 1, 2, 3])
		# transform is disabled for now
		if self.is_transform:
			img, lbl = self.transform(img, lbl)
		log('The maximum value of img is {}'.format(np.max(img)))
		log('The unique values of lbl_patch {}'.format(np.unique(lbl)))
		# convert numpy type into torch type
		img_patch = torch.from_numpy(img_patch).float()
		lbl_patch = torch.from_numpy(lbl_patch).long()

		########### Previous data loading: End

		########### Brats17 official data loading: Begin
		# Step One
		# margin = 5
		# log('The shape of lbl is {}'.format(lbl.shape))
		# log('Step One : The unique values of lbl is {}'.format(np.unique(lbl)))
		# bbmin, bbmax = get_ND_bounding_box(flair_img, margin)
		# log('bbmin is {} and bbmax is {}'.format(bbmin, bbmax))
		# lbl_crop = crop_ND_volume_with_bounding_box(lbl, bbmin, bbmax)
		# log('Step One : The unique values of lbl_crop is {}'.format(np.unique(lbl_crop)))
		# log('Step One : The shape of lbl_crop is {}'.format(lbl_crop.shape))
		# t1_crop = crop_ND_volume_with_bounding_box(t1_img, bbmin, bbmax)
		# t1ce_crop = crop_ND_volume_with_bounding_box(t1ce_img, bbmin, bbmax)
		# t2_crop = crop_ND_volume_with_bounding_box(t2_img, bbmin, bbmax)
		# flair_crop = crop_ND_volume_with_bounding_box(flair_img, bbmin, bbmax)
		# log('The image size after crop_ND_volume_with_bounding_box is {}'.format(lbl_crop.shape))
		#
		# # Step Two
		# volume_shape = lbl_crop.shape
		# sub_label_shape = [64, 64, 64]
		# batch_sample_model = ('full', 'valid', 'valid')
		# log('batch_sample_model is {}'.format(batch_sample_model[0]))
		# boundingbox = None
		# center_point = get_random_roi_sampling_center(volume_shape, sub_label_shape, batch_sample_model, boundingbox)
		# log('The center point is {}'.format(center_point))
		# # print('The centerpoint is ', center_point)
		#
		# # Step Three
		# lbl_patch = extract_roi_from_volume(lbl_crop, center_point, sub_label_shape, 'zero')
		# log('Step Three : The unique values of lbl_patch is {}'.format(np.unique(lbl_patch)))
		# log('Step Three : The shape of lbl_patch is {}'.format(lbl_patch.shape))
		# t1_patch = extract_roi_from_volume(t1_crop, center_point, sub_label_shape, 'zero')
		# t1ce_patch = extract_roi_from_volume(t1ce_crop, center_point, sub_label_shape, 'zero')
		# t2_patch = extract_roi_from_volume(t2_crop, center_point, sub_label_shape, 'zero')
		# flair_patch = extract_roi_from_volume(flair_crop, center_point, sub_label_shape, 'zero')
		# img_patch = np.stack((t1_patch, t1ce_patch, t2_patch, flair_patch))
		#
		# # Step Four
		# lbl_patch = convert_label(in_volume=lbl_patch, label_convert_source=[0, 1, 2, 4],
		# 						  label_convert_target=[0, 1, 2, 3])
		# log('The maximum value of img is {}'.format(np.max(img_patch)))
		# log('The unique values of lbl_patch {}'.format(np.unique(lbl_patch)))
		# # img_patch = np.array(img_patch, dtype=np.uint8)
		# lbl_patch = np.array(lbl_patch, dtype=np.int32)
		# img_patch = torch.from_numpy(img_patch).float()
		# lbl_patch = torch.from_numpy(lbl_patch).long()
		# log('The size of lbl_patch is {}'.format(lbl_patch.shape))
		# log('The size of img_patch is {}'.format(img_patch.shape))
		########### Brats17 official data loading: End
		# print('I am confused')
		return img_patch, lbl_patch

	# return img_patch, lbl_patch

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
