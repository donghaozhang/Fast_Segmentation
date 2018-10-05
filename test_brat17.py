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
import argparse
from torch.autograd import Variable

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

def test_brats17(args):

	# The path of file containing the names of images required to be tested
	test_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/brats17/config17/test_names_36.txt'

	# The path of dataset
	root_path = '/home/donghao/Desktop/donghao/brain_segmentation/brain_data_full'

	# The path of the model
	model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_5.pkl'

	text_file = open(test_names_path, "r")

	lines = text_file.readlines()
	img_num = np.random.randint(0, 33)
	log('The current image number is {}'.format(img_num))
	cur_im_name = lines[img_num]
	cur_im_name = cur_im_name.replace("\n", "")
	# print('I am so confused', os.path.basename(cur_im_name))
	# print('the name after splitting is ', cur_im_name.split("|\")[0])
	img_path = root_path + '/' + cur_im_name + '/' + os.path.basename(cur_im_name)

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
	log('The shape of img is {}'.format(img.shape))
	img = np.expand_dims(img, axis=0)
	log('The shape of img after dim expansion is {}'.format(img.shape))

	# convert numpy type into torch type
	img = torch.from_numpy(img).float()
	log('The shape pf img is {}'.format(img.size()))
	img_patch = img[:, :, 2:154, :, :]
	img_patch = img[:, :, 2:122, 1:161, 1:161]

	# Setup Model
	model = torch.load(model_path)
	print('The im_sz is ', model)
	if torch.cuda.is_available():
		model.cuda(0)
		img = Variable(img_patch.cuda(0))
	# print(model)
	model.eval()
	test_result = model(img)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Params')
	parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl',
						help='Path to the saved model')
	parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
						help='Dataset to use [\'pascal, camvid, ade20k etc\']')
	parser.add_argument('--img_path', nargs='?', type=str, default=None,
						help='Path of the input image')
	parser.add_argument('--out_path', nargs='?', type=str, default=None,
						help='Path of the output segmap')
	args = parser.parse_args()
	test_brats17(args)
