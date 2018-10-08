import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

DEBUG = False


def log(s):
	if DEBUG:
		print(s)


def cross_entropy2d(input, target, weight=None, size_average=True):
	n, c, h, w = input.size()
	log_p = F.log_softmax(input)
	log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
	log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
	log_p = log_p.view(-1, c)
	mask = target >= 0

	target = target[mask]
	# print('log_p ', log_p.size())
	# print('target ', target.size())
	loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
	# target_test = target.view(-1, 1)
	# logpt = log_p.gather(1, target_test)
	# print('loss sum logpt', logpt.sum())
	# print('loss original', loss)

	# if size_average:
	#     loss /= mask.data.sum()
	return loss


def cross_entropy3d_new(input, target, weight=None, size_average=True):
	log('The input size is {}'.format(input.size()))
	log('The output size is {}'.format(target.size()))
	# input: (n, c, h, w, z), target: (n, h, w, z)
	n, c, h, w, z = input.size()
	# log_p: (n, c, h, w, z)
	log_p = F.log_softmax(input, dim=1)
	# log_p: (n*h*w*z, c)
	log_p = log_p.permute(0, 4, 3, 2, 1).contiguous().view(-1, c)  # make class dimension last dimension
	log_p = log_p[
		target.view(n * h * w * z, 1).repeat(1, c) >= 0]  # this looks wrong -> Should rather be a one-hot vector
	log_p = log_p.view(-1, c)
	# target: (n*h*w*z,)
	mask = target >= 0
	target = target[mask]
	loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
	return loss


def cross_entropy3d(input, target, weight=None, size_average=True):
	log('The size of input is {}'.format(input.size()))
	input = F.log_softmax(input, dim=1)
	log('LOSS=>CrossEntropy3D=>input.size():{} target.size():{}'.format(input.size(), target.size()))
	weight = torch.tensor([0.001, 1, 1, 1], dtype=torch.float)
	weight = weight.cuda()
	loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average)
	return loss(input, target)


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class CrossEntropyLoss2d(nn.Module):
	def __init__(self, weight=None, size_average=True, nclass=2):
		super(CrossEntropyLoss2d, self).__init__()
		self.nll_loss = nn.NLLLoss2d(weight, size_average)
		self.nclass = nclass

	def forward(self, inputs, targets):
		x = inputs.permute(0, 2, 3, 1)
		y = targets.permute(0, 2, 3, 1)
		x = x.resize(x.size(0) * x.size(1) * x.size(2), self.nclass)
		y = y.resize(y.size(0) * y.size(1) * y.size(2))
		return self.nll_loss(F.log_softmax(x), y)


class CrossEntropyLoss3d(nn.Module):
	def __init__(self, weight=None, size_average=True, nclass=2):
		super(CrossEntropyLoss3d, self).__init__()
		self.nll_loss = nn.NLLLoss(weight, size_average)
		self.nclass = nclass

	def forward(self, inputs, targets):
		x = inputs.permute(0, 2, 3, 4, 1)
		y = targets.permute(0, 2, 3, 4, 1)
		x = x.resize(x.size(0) * x.size(1) * x.size(2) * x.size(3), self.nclass)
		y = y.resize(y.size(0) * y.size(1) * y.size(2) * y.size(3))
		return self.nll_loss(F.log_softmax(x), y)


class FocalCrossEntropyLoss2d(nn.Module):
	'''
	Ported from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
	'''

	def __init__(self, gamma=2, alpha=0.25, size_average=True):
		super(FocalCrossEntropyLoss2d, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
		if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
		self.size_average = size_average

	def forward(self, input, target):
		log('The input is {}'.format(input.size()))
		log('The target is {}'.format(target.size()))
		if input.dim() > 2:
			input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
		target = target.view(-1, 1)

		logpt = F.log_softmax(input)
		logpt = logpt.gather(1, target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())

		if self.alpha is not None:
			if self.alpha.type() != input.data.type():
				self.alpha = self.alpha.type_as(input.data)
			at = self.alpha.gather(0, target.data.view(-1))
			logpt = logpt * Variable(at)

		loss = -1 * (1 - pt) ** self.gamma * logpt
		loss = loss.sum()
		# if self.size_average:
		# 	loss = loss.mean()
		# else:
		# 	loss = loss.sum()
		return loss


class FocalCrossEntropyLoss3d(FocalCrossEntropyLoss2d):
	# All the same
	pass


import torch
from torch.autograd import Function
from itertools import repeat
import numpy as np


# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

class DiceLoss(Function):

	def __init__(self, *args, **kwargs):
		pass

	def forward(self, input, target, save=True):
		if save:
			self.save_for_backward(input, target)
		eps = 0.000001
		_, result_ = input.max(1)
		result_ = torch.squeeze(result_)
		if input.is_cuda:
			result = torch.cuda.FloatTensor(result_.size())
			self.target_ = torch.cuda.FloatTensor(target.size())
		else:
			result = torch.FloatTensor(result_.size())
			self.target_ = torch.FloatTensor(target.size())
		result.copy_(result_)
		self.target_.copy_(target)
		target = self.target_
		#       print(input)
		intersect = torch.dot(result, target)
		# binary values so sum the same as sum of squares
		result_sum = torch.sum(result)
		target_sum = torch.sum(target)
		union = result_sum + target_sum + (2 * eps)

		# the target volume can be empty - so we still want to
		# end up with a score of 1 if the result is 0/0
		IoU = intersect / union
		print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
			union, intersect, target_sum, result_sum, 2 * IoU))
		out = torch.FloatTensor(1).fill_(2 * IoU)
		self.intersect, self.union = intersect, union
		return out

	def backward(self, grad_output):
		input, _ = self.saved_tensors
		intersect, union = self.intersect, self.union
		target = self.target_
		gt = torch.div(target, union)
		IoU2 = intersect / (union * union)
		pred = torch.mul(input[:, 1], IoU2)
		dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
		grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
								torch.mul(dDice, grad_output[0])), 0)
		return grad_input, None


def dice_loss(input, target):
	return DiceLoss()(input, target)


def dice_error(input, target):
	eps = 0.000001
	_, result_ = input.max(1)
	result_ = torch.squeeze(result_)
	if input.is_cuda:
		result = torch.cuda.FloatTensor(result_.size())
		target_ = torch.cuda.FloatTensor(target.size())
	else:
		result = torch.FloatTensor(result_.size())
		target_ = torch.FloatTensor(target.size())
	result.copy_(result_.data)
	target_.copy_(target.data)
	target = target_
	intersect = torch.dot(result, target)

	result_sum = torch.sum(result)
	target_sum = torch.sum(target)
	union = result_sum + target_sum + 2 * eps
	intersect = np.max([eps, intersect])
	# the target volume can be empty - so we still want to
	# end up with a score of 1 if the result is 0/0
	IoU = intersect / union
	#    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
	#        union, intersect, target_sum, result_sum, 2*IoU))
	return 2 * IoU
