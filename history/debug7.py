import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable
from torch import optim


class double_conv(nn.Module):
	'''(conv => BN => ReLU) * 2'''

	def __init__(self, in_ch, out_ch):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x


class inconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(inconv, self).__init__()
		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x):
		x = self.conv(x)
		return x


class down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
			nn.MaxPool2d(2),
			double_conv(in_ch, out_ch)
		)

	def forward(self, x):
		x = self.mpconv(x)
		return x


class up(nn.Module):
	def __init__(self, in_ch, out_ch, bilinear=True):
		super(up, self).__init__()

		#  would be a nice idea if the upsampling could be learned too,
		#  but my machine do not have enough memory to handle all those weights
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffX = x1.size()[2] - x2.size()[2]
		diffY = x1.size()[3] - x2.size()[3]
		x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
						diffY // 2, int(diffY / 2)))
		x = torch.cat([x2, x1], dim=1)
		x = self.conv(x)
		return x


class outconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(outconv, self).__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 1)

	def forward(self, x):
		x = self.conv(x)
		return x


class UNet(nn.Module):

	def __init__(self, n_channels, n_classes):
		super(UNet, self).__init__()
		self.inc = inconv(n_channels, 64)
		self.down1 = down(64, 128)
		self.down2 = down(128, 256)
		self.down3 = down(256, 512)
		self.down4 = down(512, 512)
		self.up1 = up(1024, 256)
		self.up2 = up(512, 128)
		self.up3 = up(256, 64)
		self.up4 = up(128, 64)
		self.outc = outconv(64, n_classes)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		return x


class DiceCoeff(Function):
	"""Dice coeff for individual examples"""

	def forward(self, input, target):
		self.save_for_backward(input, target)
		eps = 0.0001
		self.inter = torch.dot(input.view(-1), target.view(-1))
		self.union = torch.sum(input) + torch.sum(target) + eps

		t = (2 * self.inter.float() + eps) / self.union.float()
		return t

	# This function has only a single output, so it gets only one gradient
	def backward(self, grad_output):
		print('has this being called')

		input, target = self.saved_variables
		grad_input = grad_target = None

		if self.needs_input_grad[0]:
			grad_input = grad_output * 2 * (target * self.union + self.inter) \
						 / self.union * self.union
		if self.needs_input_grad[1]:
			grad_target = None

		return grad_input, grad_target


def dice_coeff(input, target):
	"""Dice coeff for batches"""
	if input.is_cuda:
		s = torch.FloatTensor(1).cuda().zero_()
	else:
		s = torch.FloatTensor(1).zero_()

	for i, c in enumerate(zip(input, target)):
		s = s + DiceCoeff().forward(c[0], c[1])

	return s / (i + 1)


fake_im_num = 1
imgs = np.random.rand(fake_im_num, 4, 256, 256)
true_masks = np.random.rand(fake_im_num, 256, 256)
imgs = torch.from_numpy(imgs).float()
true_masks = torch.from_numpy(true_masks).float()
true_masks = Variable(true_masks)

if torch.cuda.is_available():
	imgs = imgs.cuda()
	true_masks = true_masks.cuda()
net = UNet(n_channels=4, n_classes=3)
net = net.cuda()
optimizer = optim.SGD(net.parameters(),
					  lr=1e-5,
					  momentum=0.9,
					  weight_decay=0.0005)
masks_pred = net(imgs)
masks_probs = F.sigmoid(masks_pred)
print('The size of mask_probs_flat is ', masks_probs.size())
print('The size of true_masks_flat is ', true_masks.size())
masks_probs_flat = masks_probs.view(-1)
true_masks_flat = true_masks.view(-1)
criterion = DiceCoeff()
print('The requires_grad attribute is ', true_masks_flat.requires_grad)
loss = criterion(true_masks_flat, true_masks_flat)
loss = Variable(loss.data)
optimizer.zero_grad()
torch_tensor_learn = torch.zeros((1, 256, 256), requires_grad=True)
print('The size of torch_tensor_learn is {}'.format(torch_tensor_learn.size()))
loss.backward()
