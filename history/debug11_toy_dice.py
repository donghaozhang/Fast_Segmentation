# Code in file autograd/two_layer_net_autograd.py
import torch
import torch.nn as nn

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# Create random Tensors for weights; setting requires_grad=True means that we
# want to compute gradients for these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, requires_grad=True)


class DiceCoeff(nn.Module):
	"""Dice coeff for individual examples"""

	def __init__(self):
		super(DiceCoeff, self).__init__()

	def forward(self, input, target):
		# self.save_for_backward(input, target)
		eps = 0.0001
		self.inter = torch.dot(input.view(-1), target.view(-1))
		self.union = torch.sum(input) + torch.sum(target) + eps

		t = (2 * self.inter.float() + eps) / self.union.float()
		print(t)
		return t


learning_rate = 1e-6
for t in range(500):
	# Forward pass: compute predicted y using operations on Tensors. Since w1 and
	# w2 have requires_grad=True, operations involving these Tensors will cause
	# PyTorch to build a computational graph, allowing automatic computation of
	# gradients. Since we are no longer implementing the backward pass by hand we
	# don't need to keep references to intermediate values.
	y_pred = x.mm(w1).clamp(min=0).mm(w2)

	# Compute and print loss. Loss is a Tensor of shape (), and loss.item()
	# is a Python number giving its value.
	# loss = (y_pred - y).pow(2).sum()
	criterion = DiceCoeff()
	loss2 = criterion(y, y_pred)
	# print(t, loss.item())

	# Use autograd to compute the backward pass. This call will compute the
	# gradient of loss with respect to all Tensors with requires_grad=True.
	# After this call w1.grad and w2.grad will be Tensors holding the gradient
	# of the loss with respect to w1 and w2 respectively.
	# loss.backward()
	loss2.backward()

	# Update weights using gradient descent. For this step we just want to mutate
	# the values of w1 and w2 in-place; we don't want to build up a computational
	# graph for the update steps, so we use the torch.no_grad() context manager
	# to prevent PyTorch from building a computational graph for the updates
	with torch.no_grad():
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad

		# Manually zero the gradients after running the backward pass
		w1.grad.zero_()
		w2.grad.zero_()
