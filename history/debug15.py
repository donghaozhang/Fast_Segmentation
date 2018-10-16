DEBUG = False


def log(s):
	if DEBUG:
		print(s)



import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import time

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
		padding = [offset3 // 2, offset3 - offset3 // 2, offset2 // 2, offset2 - offset2 // 2, offset1 // 2,
				   offset1 - offset1 // 2]
		log('>>>unetUp3d_regression_res: padding is: {}'.format(padding))
		outputs1 = F.pad(inputs1, padding)
		log('>>>unetUp3d_regression_res: after padding inputs1, we get outputs1 size {}'.format(outputs1.size()))

		output = torch.cat([outputs1, outputs2], 1)
		log('>>>unetUp3d_regression_res: after cat outputs1 and outputs2: {}'.format(output.size()))

		output = self.conv(output)
		log('>>>unetUp3d_regression_res: after conv: {}'.format(output.size()))
		return output


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


class unet3dregSmartStudentRes(nn.Module):
	def __init__(
			self,
			feature_scale=4,
			n_classes=1,
			is_deconv=True,
			in_channels=1,
			is_batchnorm=True
	):
		super(unet3dregSmartStudentRes, self).__init__()
		self.is_deconv = is_deconv
		self.in_channels = in_channels
		self.is_batchnorm = is_batchnorm
		self.feature_scale = feature_scale

		# filters = [64, 128, 256, 512, 1024]
		filters = [64, 128, 256]  # 16, 32, 64
		filters = [int(x / self.feature_scale) for x in filters]

		# downsampling
		self.conv1 = unetConv2_3d_regression(self.in_channels, filters[0], self.is_batchnorm)
		self.smartresConv1 = unetResConv_3d(filters[0], filters[0], self.is_batchnorm)
		self.maxpool1 = nn.MaxPool3d(kernel_size=2)
		# 1x1 convolutions are used to compute reductions before the expensive 3x3 convolutions
		self.conv_mid = nn.Conv3d(filters[0], filters[1], kernel_size=1)
		self.maxpool2 = nn.MaxPool3d(kernel_size=2)
		self.conv3 = unetConv2_3d_regression(filters[1], filters[2], self.is_batchnorm)
		self.smartresConv2 = unetResConv_3d(filters[2], filters[2], self.is_batchnorm)

		# upsampling
		self.up_concat2 = unetUp3d_regression_res(filters[2], filters[1], self.is_deconv)
		self.up_concat1 = unetUp3d_regression_res(filters[1], filters[0], self.is_deconv)

		# final conv (without any concat)
		self.smartFinal = nn.Conv3d(filters[0], n_classes, 1)

		self.smartTanh = nn.Tanh()

	def forward(self, inputs):
		# log('unet3dregStudent: inputs size is {}'.format(inputs.size()))
		# conv1 = self.conv1(inputs)
		# log('unet3dregStudent: after conv1 size is {}'.format(conv1.size()))
		# maxpool1 = self.maxpool1(conv1)
		# log('unet3dregStudent: after maxpool1 size is {}'.format(maxpool1.size()))
		# conv_mid = self.conv_mid(maxpool1)
		# log('unet3dregStudent: after conv_mid size is {}'.format(conv_mid.size()))
		# maxpool2 = self.maxpool2(conv_mid)
		# log('unet3dregStudent: after maxpool2 size is {}'.format(maxpool2.size()))
		#
		# conv3 = self.conv3(maxpool2)
		# log('unet3dregStudent: after conv3 size is {}'.format(conv3.size()))
		#
		#
		#
		# up2 = self.up_concat2(conv_mid, conv3)
		# log('unet3dregStudent: after cat conv2 and conv_mid => up2 {}'.format(up2.size()))
		# up1 = self.up_concat1(conv1, up2)
		# log('unet3dregStudent: after cat conv1 and up2 => up1 {}'.format(up1.size()))
		#
		# final = self.smartFinal(up1)
		# log('unet3dregStudent: after final conv  => final {}'.format(final.size()))
		#
		# final = self.smartTanh(final)
		# log('unet3dregStudent: after tanh  => final {}'.format(final.size()))
		log('unet3dregStudent: inputs size is {}'.format(inputs.size()))
		conv1 = self.conv1(inputs)
		log('unet3dregStudent: after conv1 size is {}'.format(conv1.size()))
		resconv1 = self.smartresConv1(conv1)
		log('unet3dregStudent: after resconv1 size is {}'.format(resconv1.size()))
		maxpool1 = self.maxpool1(resconv1)
		log('unet3dregStudent: after maxpool1 size is {}'.format(maxpool1.size()))
		conv_mid = self.conv_mid(maxpool1)
		log('unet3dregStudent: after conv_mid size is {}'.format(conv_mid.size()))
		maxpool2 = self.maxpool2(conv_mid)
		log('unet3dregStudent: after maxpool2 size is {}'.format(maxpool2.size()))

		conv3 = self.conv3(maxpool2)
		log('unet3dregStudent: after conv3 size is {}'.format(conv3.size()))
		resconv2 = self.smartresConv2(conv3)
		log('unet3dregStudent: after resconv2 size is {}'.format(resconv2.size()))

		up2 = self.up_concat2(conv_mid, resconv2)
		log('unet3dregStudent: after cat conv2 and conv_mid => up2 {}'.format(up2.size()))
		up1 = self.up_concat1(conv1, up2)
		log('unet3dregStudent: after cat conv1 and up2 => up1 {}'.format(up1.size()))

		final = self.smartFinal(up1)
		log('unet3dregStudent: after final conv  => final {}'.format(final.size()))

		final = self.smartTanh(final)
		log('unet3dregStudent: after tanh  => final {}'.format(final.size()))

		return final, conv1, conv3, up2, up1


#print("Number of ticks since 12:00am, January 1, 1970:", ticks)
fake_im_num = 1
unet_model_3D = unet3dregSmartStudentRes(feature_scale=4, n_classes=4, is_deconv=True, in_channels=1)
unet_model_3D.cuda()
numpy_fake_image_3d = np.random.rand(fake_im_num, 1sta, 8, 160, 160)
tensor_fake_image_3d = torch.FloatTensor(numpy_fake_image_3d)
torch_fake_image_3d = Variable(tensor_fake_image_3d).cuda()
ticks = time.time()
output_3d = unet_model_3D(torch_fake_image_3d)
# log(".........")
ticks_two = time.time()
print(ticks_two-ticks)