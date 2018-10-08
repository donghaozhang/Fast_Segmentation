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

DEBUG = True


def log(s):
	if DEBUG:
		print(s)


def save_array_as_nifty_volume(data, filename, reference_name = None):
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
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)

def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume):
	"""
	set a subregion to an nd image.
	"""
	dim = len(bb_min)
	out = volume
	if (dim == 2):
		out[np.ix_(range(bb_min[0], bb_max[0] + 1),
				   range(bb_min[1], bb_max[1] + 1))] = sub_volume
	elif (dim == 3):
		out[np.ix_(range(bb_min[0], bb_max[0] + 1),
				   range(bb_min[1], bb_max[1] + 1),
				   range(bb_min[2], bb_max[2] + 1))] = sub_volume
	elif (dim == 4):
		out[np.ix_(range(bb_min[0], bb_max[0] + 1),
				   range(bb_min[1], bb_max[1] + 1),
				   range(bb_min[2], bb_max[2] + 1),
				   range(bb_min[3], bb_max[3] + 1))] = sub_volume
	else:
		raise ValueError("array dimension should be 2, 3 or 4")
	return out


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


def test_brats17():
	# The path of file containing the names of images required to be tested
	# test_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/brats17/config17/test_names_36.txt'
	test_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/brats17/config17/train_names_all.txt'
	# The path of dataset
	root_path = '/home/donghao/Desktop/donghao/brain_segmentation/brain_data_full'

	# The path of the model
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_140.pkl'
	model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_255.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/unet3d_brats17_loader_1_110.pkl'
	log('dirname is {}'.format(os.path.dirname(model_path)))
	model_basename = os.path.basename(model_path)
	log('The basename is {}'.format(os.path.basename(model_path)))
	model_basename_no_ext = os.path.splitext(model_basename)[0]
	log('The model_basename_no_ext is {}'.format(model_basename_no_ext))
	if not os.path.exists('runs/'+model_basename_no_ext):
		os.makedirs('runs/'+model_basename_no_ext)
	text_file = open(test_names_path, "r")

	lines = text_file.readlines()
	log('The number of images is {}'.format(len(lines)))
	img_voxel_num= 240*240*150
	zero_pt_total = 0
	one_pt_total = 0
	two_pt_total = 0
	four_pt_total = 0
	for i in range(0, len(lines)):
		# img_num = np.random.randint(0, len(lines))
		img_num = i
		log('The current image number is {}'.format(img_num))
		cur_im_name = lines[img_num]
		cur_im_name = cur_im_name.replace("\n", "")
		# print('I am so confused', os.path.basename(cur_im_name))
		# print('the name after splitting is ', cur_im_name.split("|\")[0])
		img_path = root_path + '/' + cur_im_name + '/' + os.path.basename(cur_im_name)


		# segmentation label
		lbl_path = img_path + '_seg.nii.gz'
		lbl = load_nifty_volume_as_array(filename=lbl_path, with_header=False)
		lbl_flat = lbl.flatten()
		one_count = list(lbl_flat).count(1)
		two_count = list(lbl_flat).count(2)
		# three_count = list(lbl_flat).count(3)
		four_count = list(lbl_flat).count(4)
		zero_count = -one_count-two_count-four_count
		zero_pt = img_voxel_num/zero_count
		one_pt = img_voxel_num/one_count
		two_pt  = img_voxel_num/two_count
		four_pt = img_voxel_num/four_count
		zero_pt_final = zero_pt / (zero_pt + one_pt + two_pt + four_pt)
		one_pt_final = one_pt / (zero_pt + one_pt + two_pt + four_pt)
		two_pt_final =two_pt / (zero_pt + one_pt + two_pt + four_pt)
		four_pt_final = four_pt / (zero_pt + one_pt + two_pt + four_pt)
		zero_pt_total = zero_pt_total + zero_pt_final
		one_pt_total = one_pt_total + one_pt_final
		two_pt_total = two_pt_total + two_pt_final

		log('The size of lbl is {}'.format(lbl.shape))
		log(lbl_path)
		four_pt_total = four_pt_total + four_pt_final
	zero_pt_total = zero_pt_total / len(lines)
	one_pt_total = one_pt_total / len(lines)
	two_pt_total = two_pt_total / len(lines)
	four_pt_total = four_pt_total / len(lines)
	log('lbl one count final is {}'.format(zero_pt_total))
	log('lbl two count is {}'.format(two_pt_total))
	log('lbl four count is {}'.format(four_pt_total))
	log('lbl zero count is {}'.format(zero_pt_total))
test_brats17()

