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


def normalize_try(img, mask):
	mean=np.mean(img[mask != 0])
	std=np.std(img[mask != 0])
	return (img-mean)/std

def test_brats17(args):
	# The path of file containing the names of images required to be tested
	# test_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/brats17/config17/test_names_36.txt'
	# test_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/test_names_40_hgg.txt'
	test_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/train_names_87_hgg.txt'
	# test_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/brats17/config17/train_names_all.txt'
	# test_names_path = '/home/donghao/Desktop/donghao/isbi2019/code/brats17/config17/train_names_66.txt'
	# The path of dataset
	root_path = '/home/donghao/Desktop/donghao/brain_segmentation/brain_data_full'

	# The path of the model
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_19.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_98.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_140.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_255.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_255.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_126_4455.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_995_5624.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_995_1429.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_900_4144.pkl' # Is this current best? I kind of forgot, but it should be fine
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/unet3d_brats17_loader_1_500.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/unet3d_brats17_loader_1_19.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/unet3d_brats17_loader_1_85.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/unet3d_brats17_loader_1_99.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/unet3d_cls_brats17_loader_1_121.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_93_9875.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_9_8591_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_57_1648.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_294_3192_min.pkl' # Current best
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/unet3d_cls_brats17_loader_1_288_4130_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_294_9911_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_251_3020_min.pkl' # best best
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_280_6470_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/7466/bisenet3Dbrain_brats17_loader_1_185.pkl' # batch size 1 lr e-2
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/1918/bisenet3Dbrain_brats17_loader_1_263_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/1918/bisenet3Dbrain_brats17_loader_1_2991_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/2095/bisenet3Dbrain_brats17_loader_1_273_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/2177/bisenet3Dbrain_brats17_loader_1_293_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/3616/bisenet3Dbrain_brats17_loader_1_240_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/1108/bisenet3Dbrain_brats17_loader_1_475_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/9863/FCDenseNet57_brats17_loader_1_33.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/9863/FCDenseNet57_brats17_loader_1_599.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/9863/FCDenseNet57_brats17_loader_1_420_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/105/FCDenseNet57_brats17_loader_1_599.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/105/FCDenseNet57_brats17_loader_1_323_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/2779/FCDenseNet57_brats17_loader_1_599.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/9863/FCDenseNet57_brats17_loader_1_599.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/9667/FCDenseNet57_brats17_loader_1_599.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/3436/bisenet3Dbrain_brats17_loader_1_1145_min.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/3837/FCDenseNet57_brats17_loader_1_14.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/3683/FCDenseNet57_brats17_loader_1_140.pkl'
	# model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/3683/FCDenseNet57_brats17_loader_1_420.pkl'
	model_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/5926/FCDenseNet57_brats17_loader_1_390.pkl'
	log('dirname is {}'.format(os.path.dirname(model_path)))
	model_basename = os.path.basename(model_path)
	# print('xxx', os.path.basename(os.path.dirname(model_path)))
	log('The basename is {}'.format(os.path.basename(model_path)))
	model_basename_no_ext = os.path.splitext(model_basename)[0]
	# print(model_path.split)
	log('The model_basename_no_ext is {}'.format(model_basename_no_ext))
	log_number = os.path.basename(os.path.dirname(model_path))
	if not os.path.exists('runs/'+log_number +'/'+model_basename_no_ext):
		os.makedirs('runs/'+ log_number +'/'+ model_basename_no_ext)
	text_file = open(test_names_path, "r")

	lines = text_file.readlines()
	log('The number of images is {}'.format(len(lines)))
	for i in range(0, len(lines)):
		img_num = i
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

		# t1_img = normalize_try(t1_img, lbl)
		# t1ce_img = normalize_try(t1ce_img, lbl)
		# t2_img = normalize_try(t2_img, lbl)
		# flair_img = normalize_try(flair_img, lbl)

		img = np.stack((t1_img, t2_img, t1ce_img, flair_img))
		input_im_sz = img.shape
		log('The shape of img is {}'.format(img.shape))
		img = np.expand_dims(img, axis=0)
		log('The shape of img after dim expansion is {}'.format(img.shape))

		# convert numpy type into torch type
		img = torch.from_numpy(img).float()
		log('The shape pf img is {}'.format(img.size()))

		# Setup Model
		model = torch.load(model_path)
		if torch.cuda.is_available():
			model.cuda(0)
		# print(model)
		model.eval()
		final_label = np.zeros([input_im_sz[1], input_im_sz[2], input_im_sz[3]], np.int16)
		shapeX = input_im_sz[1]
		shapeY = input_im_sz[2]
		shapeZ = input_im_sz[3]
		patch_size = [64, 64, 64]
		log('The patch_size is {} {} {}'.format(patch_size[0], patch_size[1], patch_size[2]))
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
					if torch.cuda.is_available():
						img_patch = Variable(img_patch.cuda(0))
					# print('patch tensor size: {}'.format(patch.size()))
					pred = model(img_patch)

					pred = np.squeeze(pred.data.cpu().numpy(), axis=0)
					pred = np.argmax(pred, axis=0)

					## The Unet regression start
					# Convert CUDA Tensor to Numpy
					# pred = pred.to(torch.device("cpu"))
					# pred = pred.detach().numpy()
					# log('The maximum value of final label is {}'.format(pred.max()))
					# log('The minimum value of final label is {}'.format(pred.min()))
					# log('The unique values are {}'.format(np.unique(pred)))
					# log('The length of unique values is {}'.format(len(np.unique(pred))))
					# pred = (pred-pred.min())/(pred.max()-pred.min()) * 1000
					# log('The maximum value of final label is {}'.format(pred.max()))
					# log('The minimum value of final label is {}'.format(pred.min()))
					# # log('The unique values are {}'.format(np.unique(pred)))
					# log('The length of unique values is {}'.format(len(np.unique(pred))))
					## The Unet regression end

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
		# log('The maximum value of final label is {}'.format(final_label.max()))
		# log('The minimum value of final label is {}'.format(final_label.min()))
		final_label = convert_label(in_volume=final_label, label_convert_source=[0, 1, 2, 3],
								label_convert_target=[0, 1, 2, 4])
		bi_t1_img = t1_img > 0
		final_label = final_label * bi_t1_img
		# print('The values of this prediction is {}'.format(np.unique(final_label)))
		save_array_as_nifty_volume(final_label, 'runs/' + log_number + '/' + model_basename_no_ext + '/' + os.path.basename(cur_im_name) + ".nii.gz")





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
