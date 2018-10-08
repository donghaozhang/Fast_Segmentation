import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


from torch.autograd import Variable
from torch.utils import data

# get_model is defined in the __init__.py file
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d, cross_entropy3d, FocalCrossEntropyLoss3d, FocalCrossEntropyLoss2d, DiceLoss
from ptsemseg.metrics import scores
from lr_scheduling import *
# from tensorboardX import SummaryWriter

DEBUG = False


def log(s):
	if DEBUG:
		print(s)

def train(args):
	# Setup Dataloader
	print('###### Step One: Setup Dataloader')
	data_loader = get_loader(args.dataset)
	data_path = get_data_path(args.dataset)
	# writer = SummaryWriter()

	# For 2D dataset keep is_transform True
	# loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))

	# For 3D dataset keep is_transform False
	loader = data_loader(data_path, is_transform=False, img_size=(args.img_rows, args.img_cols))

	n_classes = loader.n_classes
	trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

	# Setup Model
	print('###### Step Two: Setup Model')
	model = get_model(args.arch, n_classes)

	if torch.cuda.is_available():
		model.cuda(0)
		test_image, test_segmap = loader[0]
		test_image = Variable(test_image.unsqueeze(0).cuda(0))
	else:
		test_image, test_segmap = loader[0]
		test_image = Variable(test_image.unsqueeze(0))

	log('The optimizer is Adam')
	log('The learning rate is {}'.format(args.l_rate))
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
	# optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.99)
	# optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate)

	# Train Model
	print('###### Step Three: Training Model')
	for epoch in range(args.n_epoch):
		img_counter = 1
		loss_sum = 0
		for i, (images, labels) in enumerate(trainloader):
			img_counter = img_counter + 1
			if torch.cuda.is_available():
				images = Variable(images.cuda(0))
				labels = Variable(labels.cuda(0))
			else:
				images = Variable(images)
				labels = Variable(labels)

			optimizer.zero_grad()
			log('The maximum value of input image is {}'.format(images.max()))
			outputs = model(images)
			if args.arch == 'bisenet3Dbrain':
				loss = cross_entropy3d(outputs, labels)
			elif args.arch == 'unet3d':
				# criterion = DiceLoss()
				# loss = criterion(outputs, labels)
				log('The unique value of labels are {}'.format(np.unique(labels)))
				log('The maximum of outputs are {}'.format(outputs.max()))
				log('The size of output is {}'.format(outputs.size()))
				log('The size of labels is {}'.format(labels.size()))
				loss = cross_entropy3d(outputs, labels)
				#criterion = FocalCrossEntropyLoss3d()
				#loss = criterion(outputs, labels)
			else:
				loss = cross_entropy2d(outputs, labels)

			loss.backward()
			optimizer.step()
			loss_sum = loss_sum + torch.Tensor([loss.data]).unsqueeze(0).cpu()

		avg_loss = loss_sum / img_counter
		avg_loss_array = np.array(avg_loss)
		print('The current loss of epoch', epoch, 'is', avg_loss_array[0][0])

		if epoch % 5 == 0:
			torch.save(model, "runs/{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch))
		# training model will be saved
		# writer.add_scalar('train_main_loss', avg_loss_array[0][0], epoch)
		# test_output = model(test_image)
		# predicted = loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
		# target = loader.decode_segmap(test_segmap.numpy())
		# torch.save(model, "runs/{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Hyperparams')
	parser.add_argument('--arch', nargs='?', type=str, default='fcn8s',
						help='Architecture to use [\'fcn8s, unet, segnet etc\']')
	parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
						help='Dataset to use [\'pascal, camvid, ade20k etc\']')
	parser.add_argument('--img_rows', nargs='?', type=int, default=256,
						help='Height of the input image')
	parser.add_argument('--img_cols', nargs='?', type=int, default=256,
						help='Height of the input image')
	parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
						help='# of the epochs')
	parser.add_argument('--batch_size', nargs='?', type=int, default=1,
						help='Batch Size')
	parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
						help='Learning Rate')
	parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
						help='Divider for # of features to use')
	args = parser.parse_args()
	train(args)
