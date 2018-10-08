import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.system('echo $CUDA_VISIBLE_DEVICES')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class TransientModel(nn.Module):
	def __init__(self):
		super(TransientModel, self).__init__()
		self.conv1 = nn.Conv2d(16, 8, kernel_size=1)
		self.conv2 = nn.Conv2d(8, 4, kernel_size=1)
		self.conv3 = nn.Conv2d(4, 2, kernel_size=1)
		self.conv4 = nn.Conv2d(2, 1, kernel_size=1)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		return x


class MyLoss(nn.Module):
	def __init__(self):
		super(MyLoss, self).__init__()

	def forward(self, pred, truth):
		truth = torch.mean(truth, 1)
		truth = truth.view(-1, 2048)
		pred = pred.view(-1, 2048)
		return torch.mean(torch.mean((pred - truth) ** 2, 1), 0)


class MyTrainData(data.Dataset):
	def __init__(self):
		self.video_path = '/data/FrameFeature/Penn/'
		self.video_file = '/data/FrameFeature/Penn_train.txt'
		fp = open(self.video_file, 'r')
		lines = fp.readlines()
		fp.close()
		self.video_name = []
		for line in lines:
			self.video_name.append(line.strip().split(' ')[0])

	def __len__(self):
		return len(self.video_name)

	def __getitem__(self, index):
		data = load_feature(os.path.join(self.video_path, self.video_name[index]))
		data = np.expand_dims(data, 2)
		return data


def train(model, train_loader, myloss, optimizer, epoch):
	model.train()
	for batch_idx, train_data in enumerate(train_loader):
		train_data = Variable(train_data).cuda()
		optimizer.zero_grad()
		output = model(train_data)
		loss = myloss(output, train_data)
		loss.backward()
		optimizer.step()
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
				epoch, batch_idx * len(train_data), len(train_loader.dataset),
					   100. * batch_idx / len(train_loader), loss.data.cpu().numpy()[0]))


def main():
	model = TransientModel().cuda()
	myloss = MyLoss()

	train_data = MyTrainData()
	train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)

	optimizer = optim.SGD(model.parameters(), lr=0.001)

	for epoch in range(10):
		train(model, train_loader, myloss, optimizer, epoch)


if __name__ == '__main__':
	main()
