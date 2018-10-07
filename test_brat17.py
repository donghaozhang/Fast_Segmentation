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


def test_brats17(args):
	# The path of file containing the names of images required to be tested
	test_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/brats17/config17/test_names_36.txt'

	# The path of dataset
	root_path = '/home/donghao/Desktop/donghao/brain_segmentation/brain_data_full'

	# The path of the model
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_140.pkl'
	model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/unet3d_brats17_loader_1_1000.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_255.pkl'
	text_file = open(test_names_path, "r")

	lines = text_file.readlines()
	print('The number of images is ', )
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
	input_im_sz = img.shape
	log('The shape of img is {}'.format(img.shape))
	img = np.expand_dims(img, axis=0)
	log('The shape of img after dim expansion is {}'.format(img.shape))

	# convert numpy type into torch type
	img = torch.from_numpy(img).float()
	log('The shape pf img is {}'.format(img.size()))
	img_patch = img[:, :, 2:154, 90:154, 90:154]
	# img_patch = img[:, :, 2:122, 1:161, 1:161]

	# Setup Model
	model = torch.load(model_path)
	if torch.cuda.is_available():
		model.cuda(0)
		img_patch = Variable(img_patch.cuda(0))
	# print(model)
	model.eval()
	test_result = model(img_patch)
	log('test result size: {}'.format(test_result.size()))
	pred = np.squeeze(test_result.data.cpu().numpy(), axis=0)
	log('The shape of pred after squeeze is {}'.format(pred.shape))
	pred = np.argmax(pred, axis=0)
	log('The value of pred is '.format(np.unique(pred)))
	log('The shape of pred is {}'.format(pred.shape))
	temp_size = input_im_sz
	final_label = np.zeros([temp_size[1], temp_size[2], temp_size[3]], np.int16)
	#final_label[2:154, 90:154, 90:154] = pred

	# print('The test_result is ', test_result.size())

	log('apply 160*160*8 network to the img')
	shapeX = temp_size[1]
	shapeY = temp_size[2]
	shapeZ = temp_size[3]
	patch_size = [64, 64, 64]
	stack_alongX = None
	stack_alongY = None
	stack_alongZ = None
	overlapX = 0
	overlapY = 0
	overlapZ = 0
	x = 0
	y = 0
	z = 0
	while x < shapeX:

		# residual
		if x + patch_size[0] > shapeX:
			overlapX = x - (shapeX - patch_size[0])
			x = shapeX - patch_size[0]

		y = 0
		while y < shapeY:
			# residual
			if y + patch_size[1] > shapeY:
				overlapY = y - (shapeY - patch_size[1])
				y = shapeY - patch_size[1]
			# log('overlapY: {}'.format(overlapY))

			z = 0
			while z < shapeZ:
				# residual check
				if z + patch_size[2] > shapeZ:
					overlapZ = z - (shapeZ - patch_size[2])
					z = shapeZ - patch_size[2]

				# log('overlapZ: {}'.format(overlapZ))
				img_patch = img[:, :, x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]]
				img_patch = Variable(img_patch.cuda(0))
				# print('patch tensor size: {}'.format(patch.size()))
				pred = model(img_patch)
				pred = np.squeeze(pred.data.cpu().numpy(), axis=0)
				pred = np.argmax(pred, axis=0)
				final_label[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]] = pred

				if overlapZ:
					pred = pred[:, :, overlapZ:]
					stack_alongZ = np.concatenate((stack_alongZ, pred), axis=2)
					overlapZ = 0
				else:
					if stack_alongZ is None:
						stack_alongZ = pred
					else:
						stack_alongZ = np.concatenate((stack_alongZ, pred), axis=2)
				# log('===>z ({}/{}) loop: stack_alongZ shape: {}'.format(z, shapeZ, stack_alongZ.shape))
				z += patch_size[2]

			if overlapY:
				stack_alongZ = stack_alongZ[:, overlapY:, :]
				stack_alongY = np.concatenate((stack_alongY, stack_alongZ), axis=1)
				overlapY = 0
			else:
				if stack_alongY is None:
					stack_alongY = stack_alongZ
				else:
					stack_alongY = np.concatenate((stack_alongY, stack_alongZ), axis=1)
			# log('==>y ({}/{}) loop: stack_alongY shape: {}'.format(y, shapeY, stack_alongY.shape))
			stack_alongZ = None
			y += patch_size[1]

		if overlapX:
			stack_alongY = stack_alongY[overlapX:, :, :]
			stack_alongX = np.concatenate((stack_alongX, stack_alongY), axis=0)
			overlapX = 0
		else:
			if stack_alongX is None:
				stack_alongX = stack_alongY
			else:
				stack_alongX = np.concatenate((stack_alongX, stack_alongY), axis=0)
		# log('=>x ({}/{}) loop: stack_alongX shape: {}'.format(x, shapeX, stack_alongX.shape))
		stack_alongY = None
		x += patch_size[0]

	save_array_as_nifty_volume(final_label, "/home/donghao/Desktop/donghao/test.nii.gz")





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
