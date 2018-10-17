import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
DEBUG = False


def log(s):
	if DEBUG:
		print(s)


class SeparableConv3d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
		super(SeparableConv3d, self).__init__()

		self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
		self.pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pointwise(x)
		return x


class Block3D(nn.Module):
	def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
		super(Block3D, self).__init__()

		if out_filters != in_filters or strides != 1:
			self.skip = nn.Conv3d(in_filters, out_filters, 1, stride=strides, bias=False)
			self.skipbn = nn.BatchNorm3d(out_filters)
		else:
			self.skip = None

		self.relu = nn.ReLU(inplace=True)
		rep = []

		filters = in_filters
		if grow_first:
			rep.append(self.relu)
			rep.append(SeparableConv3d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
			rep.append(nn.BatchNorm3d(out_filters))
			filters = out_filters

		for i in range(reps - 1):
			rep.append(self.relu)
			rep.append(SeparableConv3d(filters, filters, 3, stride=1, padding=1, bias=False))
			rep.append(nn.BatchNorm3d(filters))

		if not grow_first:
			rep.append(self.relu)
			rep.append(SeparableConv3d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
			rep.append(nn.BatchNorm3d(out_filters))

		if not start_with_relu:
			rep = rep[1:]
		else:
			rep[0] = nn.ReLU(inplace=False)

		if strides != 1:
			rep.append(nn.MaxPool3d(3, strides, 1))
		self.rep = nn.Sequential(*rep)

	def forward(self, inp):
		x = self.rep(inp)

		if self.skip is not None:
			skip = self.skip(inp)
			skip = self.skipbn(skip)
		else:
			skip = inp

		x += skip
		return x


class SpatialPath3DModule(nn.Module):
	def __init__(self, conv_in_channels, conv_out_channels):
		super(SpatialPath3DModule, self).__init__()
		self.conv1 = nn.Conv3d(in_channels=conv_in_channels, out_channels=conv_out_channels, stride=2, kernel_size=3, padding=1)
		self.bn = nn.BatchNorm3d(num_features=conv_out_channels)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn(x)
		out = self.relu(x)
		return out


class AttentionRefinement3DModule(nn.Module):
	def __init__(self, conv_in_channels, conv_out_channels, pool_size):
		super(AttentionRefinement3DModule, self).__init__()
		self.conv = nn.Conv3d(in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=1, stride=1, padding=0)
		self.bn = nn.BatchNorm3d(num_features=conv_out_channels)
		self.sigmod = nn.Sigmoid()
		self.pool_size = pool_size

	def forward(self, x):
		input = x
		# print('the input size of ARM is ', x.size())

		# Convert type from float to int
		log('The value of pool_size is {}'.format(self.pool_size))
		x = F.adaptive_avg_pool3d(x, (self.pool_size[0], self.pool_size[1], self.pool_size[2]))
		log('the input size of ARM after adaptive average pooling is '.format(x.size()))
		x = self.conv(x)
		x = self.bn(x)
		x = self.sigmod(x)
		log('debug purpose the size of input is {}'.format(input.size()))
		log('the size of x is {}'.format(x.size()))
		x = torch.mul(input, x)
		return x


class FeatureFusion3DModule(nn.Module):
	def __init__(self, conv_in_channels, conv_out_channels, pool_size):
		super(FeatureFusion3DModule, self).__init__()
		self.conv1 = nn.Conv3d(in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm3d(num_features=conv_out_channels)
		self.conv2 = nn.Conv3d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=1, stride=1, padding=0)
		self.conv3 = nn.Conv3d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=1, stride=1, padding=0)
		self.pool_size = pool_size

	def forward(self, x):
		log(' debug: the size of input is {}'.format(x.size()))
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		before_mul = x

		# global pool + conv + relu + conv + sigmoid
		x = F.adaptive_avg_pool3d(x, (self.pool_size[0], self.pool_size[1], self.pool_size[2]))
		log(' debug: the size x after {}'.format(x.size()))
		x = self.conv2(x)
		x = F.relu(x, inplace=False)
		x = self.conv3(x)
		x = F.sigmoid(x)
		log('debug====> the size of before_mul is {}'.format(before_mul.size()))
		log('debug====> the size of x is {}'.format(x.size()))
		x = torch.mul(before_mul, x)
		x = x + before_mul
		return x


# My own modification starts now.
class DHUpRes(nn.Module):
	# This
	def __init__(self, conv_in_channels=4, conv_out_channels=4):
		super(DHUpRes, self).__init__()

		self.conv1 = nn.Sequential(
			nn.ConvTranspose3d(conv_in_channels, conv_out_channels, kernel_size=2, stride=2),
			nn.LeakyReLU(0.2, False),
			unetResConv_3d(conv_out_channels, conv_out_channels, False)
			)

	def forward(self, inputs):
		outputs = self.conv1(inputs)
		log('debuging....outputs are {}'.format(outputs.size()))
		log('debuging....inputs are {}'.format(inputs.size()))
		# residual - add inputs
		return outputs


class unetUp3d_regression_res(nn.Module):
	def __init__(self, in_size, out_size, is_deconv):
		super(unetUp3d_regression_res, self).__init__()
		self.conv = unetConv2_3d_regression(in_size, out_size, False)
		if is_deconv:
			log('The deconvolution is used.')
			# convTranspose with residual
			self.up = nn.Sequential(nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2),
									nn.LeakyReLU(0.2, False),
									unetResConv_3d(out_size, out_size, False))
		else:
			log('The simple interpolate is used.')
			self.up = F.interpolate(scale_factor=2, mode='bilinear')

	def forward(self, inputs1, inputs2):
		log('>>>unetUp3d_regression_res: inputs1 size {}, inputs2 size {}'.format(inputs1.size(), inputs2.size()))
		outputs2 = self.up(inputs2)
		log('>>>unetUp3d_regression_res: upsample inputs2, get outputs2 size {}'.format(outputs2.size()))
		offset1 = outputs2.size()[2] - inputs1.size()[2]
		offset2 = outputs2.size()[3] - inputs1.size()[3]
		offset3 = outputs2.size()[4] - inputs1.size()[4]
		log('>>>unetUp3d_regression_res: offset between outputs2 and inputs1 is: {}'.format(
			(offset1, offset2, offset3)))
		padding = [offset3 // 2, offset3 - offset3 // 2, offset2 // 2, offset2 - offset2 // 2, offset1 // 2, offset1 - offset1 // 2]
		log('>>>unetUp3d_regression_res: padding is: {}'.format(padding))
		outputs1 = F.pad(inputs1, padding)
		log('>>>unetUp3d_regression_res: after padding inputs1, we get outputs1 size {}'.format(outputs1.size()))

		output = torch.cat([outputs1, outputs2], 1)
		log('>>>unetUp3d_regression_res: after cat outputs1 and outputs2: {}'.format(output.size()))

		output = self.conv(output)
		log('>>>unetUp3d_regression_res: after conv: {}'.format(output.size()))
		return output


class unetConv2_3d_regression(nn.Module):
	def __init__(self, in_size, out_size, is_batchnorm):
		super(unetConv2_3d_regression, self).__init__()

		if is_batchnorm:
			self.conv1 = nn.Sequential(
				nn.Conv3d(in_size, out_size, 3, 1, 1),
				nn.BatchNorm3d(out_size),
				nn.ReLU(),
			)
			self.conv2 = nn.Sequential(
				nn.Conv3d(out_size, out_size, 3, 1, 1),
				nn.BatchNorm3d(out_size),
				nn.ReLU(),
			)
		else:
			self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, 3, 1, 1), nn.ReLU())
			self.conv2 = nn.Sequential(
				nn.Conv3d(out_size, out_size, 3, 1, 1), nn.ReLU()
			)

	def forward(self, inputs):
		outputs = self.conv1(inputs)
		outputs = self.conv2(outputs)
		return outputs


class unetResConv_3d(nn.Module):
	def __init__(self, in_size, out_size, is_batchnorm):
		super(unetResConv_3d, self).__init__()

		if is_batchnorm:
			self.conv1 = nn.Sequential(
				nn.Conv3d(in_size, out_size, 3, 1, 1),
				nn.BatchNorm3d(out_size),
				nn.ReLU(),
			)
			self.conv2 = nn.Sequential(
				nn.Conv3d(out_size, out_size, 3, 1, 1),
				nn.BatchNorm3d(out_size),
				nn.ReLU(),
			)
		else:
			self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, 3, 1, 1), nn.ReLU())
			self.conv2 = nn.Sequential(
				nn.Conv3d(out_size, out_size, 3, 1, 1), nn.ReLU()
			)

	def forward(self, inputs):
		outputs = self.conv1(inputs)
		outputs = self.conv2(outputs)
		# residual - add inputs
		outputs = torch.add(outputs, inputs)
		return outputs


class Bisenet3DBrain(nn.Module):
	"""
	Xception optimized for the ImageNet dataset, as specified in
	https://arxiv.org/pdf/1610.02357.pdf
	"""

	def __init__(self):
		""" Constructor
		Args:
			num_classes: number of classes
		"""
		super(Bisenet3DBrain, self).__init__()

		# The size input image
		self.im_sz = []

		# The size of pool size
		self.pool_size = []

		# Context Path
		self.conv1_xception39 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False)
		self.maxpool_xception39 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

		# P3
		self.block1_xception39 = Block3D(in_filters=8, out_filters=16, reps=1, strides=2, start_with_relu=True, grow_first=True)
		self.block2_xception39 = Block3D(in_filters=16, out_filters=16, reps=3, strides=1, start_with_relu=True, grow_first=True)

		# P4
		self.block3_xception39 = Block3D(in_filters=16, out_filters=32, reps=1, strides=2, start_with_relu=True, grow_first=True)
		self.block4_xception39 = Block3D(in_filters=32, out_filters=32, reps=7, strides=1, start_with_relu=True, grow_first=True)

		# P5
		self.block5_xception39 = Block3D(in_filters=32, out_filters=64, reps=1, strides=2, start_with_relu=True, grow_first=True)
		self.block6_xception39 = Block3D(in_filters=64, out_filters=64, reps=3, strides=1, start_with_relu=True, grow_first=True)

		self.arm1_context_path = AttentionRefinement3DModule(conv_in_channels=32, conv_out_channels=32, pool_size=self.pool_size)
		self.arm2_context_path = AttentionRefinement3DModule(conv_in_channels=64, conv_out_channels=64, pool_size=self.pool_size)
		# self.block3_spatial_path = AttentionRefinementModule(conv_in_channels=64, conv_out_channels=64)

		# Spatial Path
		self.block1_spatial_path = SpatialPath3DModule(conv_in_channels=4, conv_out_channels=64)
		self.block2_spatial_path = SpatialPath3DModule(conv_in_channels=64, conv_out_channels=64)
		self.block3_spatial_path = SpatialPath3DModule(conv_in_channels=64, conv_out_channels=64)

		# AVG 1 => Up block 1_1
		self.upblock1_1 = DHUpRes(conv_in_channels=32, conv_out_channels=32)

		# AVG 2 => Up block 2_1
		self.upblock2_1 = DHUpRes(conv_in_channels=32, conv_out_channels=32)
		self.upblock2_2 = DHUpRes(conv_in_channels=32, conv_out_channels=32)
		self.upblock2_3 = DHUpRes(conv_in_channels=32, conv_out_channels=32)
		#print('This is running')
		# AVG 3 => Up block 3_1
		self.upblock3_1 = DHUpRes(conv_in_channels=4, conv_out_channels=4)
		self.upblock3_2 = DHUpRes(conv_in_channels=4, conv_out_channels=4)
		self.upblock3_3 = DHUpRes(conv_in_channels=4, conv_out_channels=4)
		# self.upblock3_1 = unetUp3d_regression_res(conv_in_channels=4, conv_out_channels=4)
		self.FFM = FeatureFusion3DModule(conv_in_channels=128, conv_out_channels=4, pool_size=self.pool_size)
	# #------- init weights --------
	# for m in self.modules():
	#     if isinstance(m, nn.Conv2d):
	#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
	#         m.weight.data.normal_(0, math.sqrt(2. / n))
	#     elif isinstance(m, nn.BatchNorm2d):
	#         m.weight.data.fill_(1)
	#         m.bias.data.zero_()
	# #-----------------------------

	def forward(self, input):
		log('the size of input image is {}'.format(input.size()))
		input_size = input.size()
		self.im_sz = input.size()
		# Convert type from float to int
		input_size_x = int(input_size[2])
		input_size_y = int(input_size[3])
		input_size_z = int(input_size[4])

		avg_pool_size_x = input_size[2] / 8
		avg_pool_size_y = input_size[3] / 8
		avg_pool_size_z = input_size[4] / 8

		# Convert type
		avg_pool_size_x = int(avg_pool_size_x)
		avg_pool_size_y = int(avg_pool_size_y)
		avg_pool_size_z = int(avg_pool_size_z)

		self.pool_size = [avg_pool_size_x, avg_pool_size_y, avg_pool_size_z]

		# Context Path
		y = self.conv1_xception39(input)

		log('the size of xception39 after conv1 is {}'.format(y.size()))
		y = self.maxpool_xception39(y)
		log(' level 1: 1 / 4 the size of xception39 after maxpool is {}'.format(y.size()))

		y = self.block1_xception39(y)
		# print('the size of xception39 after block1', y.size())
		y = self.block2_xception39(y)
		log(' level 2: 1 / 8 the size of xception39 after block2 is {}'.format(y.size()))

		y = self.block3_xception39(y)
		# print(' level 3: 1 / 16 the size of xception39 after block3', y.size())
		y = self.block4_xception39(y)
		log(' level 3: 1 / 16 the size of xception39 after block4 is {}'.format(y.size()))
		# level one 256 / 2 => 112, level two 112 / 2 => 56, level three 56 / 2 => 28
		#y_16 = F.adaptive_avg_pool3d(y, (avg_pool_size_x, avg_pool_size_y, avg_pool_size_z))
		y_16 = self.upblock1_1(y)
		self.arm1_context_path.pool_size = self.pool_size
		y_arm = self.arm1_context_path(y_16)
		log(' the size of image feature after first y_arm is {}'.format(y_arm.size()))
		log('===== The required upsamling sizes are  x_{} y_{} z_{}'.format(input_size_x / y_16.size(2),
																			input_size_y / y_16.size(3),
																			input_size_z / y_16.size(4)))
		#y_16_up = self.upblock2_1(y_16)
		#y_16_up = self.upblock2_1(y_16_up)
		#y_16_up = self.upblock2_1(y_16_up)
		#log(' the size of y_16 is {}'.format(y_16.size()))
		#y_16_up = F.adaptive_avg_pool3d(y_16, (avg_pool_size_x, avg_pool_size_y, avg_pool_size_z))
		#log(' the size of y_16_up is {}'.format(y_16_up.size()))
		#y_16_up

		# Concatenate the image feature of ARM1,
		y_cat = torch.cat([y_arm, y_16], dim=1)
		# y_cat = torch.cat([y_cat, y_32_up], dim=1)
		log(' size of y_cat is {}'.format(y_cat.size()))
		# Spatial Path
		sp = self.block1_spatial_path(input)
		log(' level 1 of spatial path, the size of sp is {}'.format(sp.size()))
		sp = self.block2_spatial_path(sp)
		log(' level 2 of spatial path, the size of sp is {}'.format(sp.size()))
		sp = self.block3_spatial_path(sp)
		log(' level 3 of spatial path, the size of sp is {}'.format(sp.size()))

		# Concatenate the image feature after context path : y_cat and the image feature after spatial path : sp
		y_cat = torch.cat([y_cat, sp], dim=1)
		log(' the size of image feature after the context path and spatial path is {}'.format(y_cat.size()))
		self.FFM.pool_size = self.pool_size
		y_cat = self.FFM(y_cat)
		y_cat = self.upblock3_1(y_cat)
		y_cat = self.upblock3_2(y_cat)
		y_cat = self.upblock3_3(y_cat)
		log(' the size of image feature after FFM is {}'.format(y_cat.size()))
		log('===== The required upsamling sizes are  x_{} y_{} z_{}'.format(input_size_x/y_cat.size(2), input_size_y/y_cat.size(3), input_size_z/y_cat.size(4)))
		# y_cat = F.adaptive_avg_pool3d(y_cat, (input_size_x, input_size_y, input_size_z))
		log(' the size of image feature after FFM is {}'.format(y_cat.size()))
		return y_cat

# unet 3D brain
# log(".........")
# log('The start of 3D bisenet')
# fake_im_num = 1
# unet_model_3D = Bisenet3DBrain()
# unet_model_3D.cuda()
# numpy_fake_image_3d = np.random.rand(fake_im_num, 4, 64, 64, 64)
# tensor_fake_image_3d = torch.FloatTensor(numpy_fake_image_3d)
# torch_fake_image_3d = Variable(tensor_fake_image_3d).cuda()
# output_3d = unet_model_3D(torch_fake_image_3d)
# log(".........")
