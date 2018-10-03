import torch
import torch.nn as nn
import torch.nn.functional as F
DEBUG = True


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
		x = F.adaptive_avg_pool3d(x, (self.pool_size, self.pool_size, self.pool_size))
		# print('the input size of ARM after adaptive average pooling is ', x.size())
		x = self.conv(x)
		x = self.bn(x)
		x = self.sigmod(x)
		# print('debug purpose ', 'the size of input ', input.size(), 'the size of x is ', x.size())
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
		print(' debug: the size of input is ', x.size())
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		before_mul = x

		# global pool + conv + relu + conv + sigmoid
		x = F.adaptive_avg_pool3d(x, (self.pool_size, self.pool_size, self.pool_size))
		print(' debug: the size x after ', x.size())
		x = self.conv2(x)
		x = F.relu(x, inplace=False)
		x = self.conv3(x)
		x = F.sigmoid(x)
		# print('the size of before_mul is ', before_mul.size())
		# print('the size of x is ', x.size())
		x = torch.mul(before_mul, x)
		x = x + before_mul
		return x


class Bisenet3D(nn.Module):
	"""
	Xception optimized for the ImageNet dataset, as specified in
	https://arxiv.org/pdf/1610.02357.pdf
	"""

	def __init__(self, num_classes=1000):
		""" Constructor
		Args:
			num_classes: number of classes
		"""
		super(Bisenet3D, self).__init__()
		self.num_classes = num_classes

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

		self.fc_xception39 = nn.Linear(in_features=64, out_features=self.num_classes)

		self.arm1_context_path = AttentionRefinement3DModule(conv_in_channels=32, conv_out_channels=32, pool_size=28)
		self.arm2_context_path = AttentionRefinement3DModule(conv_in_channels=64, conv_out_channels=64, pool_size=28)
		# self.block3_spatial_path = AttentionRefinementModule(conv_in_channels=64, conv_out_channels=64)

		# Spatial Path
		self.block1_spatial_path = SpatialPath3DModule(conv_in_channels=4, conv_out_channels=64)
		self.block2_spatial_path = SpatialPath3DModule(conv_in_channels=64, conv_out_channels=64)
		self.block3_spatial_path = SpatialPath3DModule(conv_in_channels=64, conv_out_channels=64)

		self.FFM = FeatureFusion3DModule(conv_in_channels=224, conv_out_channels=2, pool_size=28)
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
		y = F.adaptive_avg_pool3d(y, (28, 28, 28))
		y_arm = self.arm1_context_path(y)
		log(' the size of image feature after first y_arm is {}'.format(y_arm.size()))

		y = self.block5_xception39(y)
		# print('the size of xception39 after block5', y.size())
		y_32 = self.block6_xception39(y)
		log(' level 4: 1 / 32 the size of xception39 after block6 is {}'.format(y.size()))
		y = F.adaptive_avg_pool3d(y_32, (28, 28, 28))
		y_arm2 = self.arm2_context_path(y)
		log(' the size of image feature after second y_arm is {}'.format(y_arm2.size()))
		y_32_up = F.adaptive_avg_pool3d(y_32, (28, 28, 28))
		log(' the size of y_32_up is '.format(y_32_up.size()))

		# Concatenate the image feature of ARM1, ARM2 and y_32_up
		y_cat = torch.cat([y_arm, y_arm2], dim=1)
		y_cat = torch.cat([y_cat, y_32_up], dim=1)
		log(' size of y_cat is '.format(y_cat.size()))
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
		y_cat = self.FFM(y_cat)
		log(' the size of image feature after FFM is {}'.format(y_cat.size()))

		y_cat = F.adaptive_avg_pool3d(y_cat, (256, 256, 256))
		log(' the size of image feature after FFM is {}'.format(y_cat.size()))
		# y = F.adaptive_avg_pool2d(y, (1, 1))
		# y = y.view(y.size(0), -1)
		# # print('the size of xception39 is ', y.size()[1])
		# y = self.fc_xception39(y)
		return y_cat


# def bisenet3D(num_classes=1000, pretrained='imagenet'):
# 	import torch
# 	model = Bisenet3D(num_classes=num_classes)
# 	if pretrained:
# 		settings = pretrained_settings['xception'][pretrained]
# 		assert num_classes == settings['num_classes'], \
# 			"num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
# 		model = Bisenet3D(num_classes=num_classes)
# 		model.load_state_dict(torch.load('/home/donghao/.torch/models/xception-squeezzed.pth'))
# 		model.input_space = settings['input_space']
# 		model.input_size = settings['input_size']
# 		model.input_range = settings['input_range']
# 		model.mean = settings['mean']
# 		model.std = settings['std']
#
# 	# # TODO: ugly
# 	# model.last_linear = model.fc
# 	# del model.fc
# 	return model
