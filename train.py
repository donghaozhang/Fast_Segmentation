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
from tensorboardX import SummaryWriter

DEBUG = True


def log(s):
	if DEBUG:
		print(s)

def train(args):
	rand_int = np.random.randint(10000)
	print('###### Step Zero: Log Number is ', rand_int)

	# Setup Dataloader
	print('###### Step One: Setup Dataloader')
	data_loader = get_loader(args.dataset)
	data_path = get_data_path(args.dataset)
	# rand_int is used to
	writer = SummaryWriter('runs/'+str(rand_int))

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
	# optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
	# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate)

	# Train Model
	print('###### Step Three: Training Model')

	epoch_loss_array_total = np.zeros([1, 2])
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
			# log('The maximum value of input image is {}'.format(images.max()))
			outputs = model(images)
			if args.arch == 'bisenet3Dbrain' or args.arch == 'unet3d_cls':
				loss = cross_entropy3d(outputs, labels)
			elif args.arch == 'unet3d_res':
				labels = labels * 40
				labels = labels + 1
				log('The unique value of labels are {}'.format(np.unique(labels)))
				log('The maximum of outputs are {}'.format(outputs.max()))
				log('The size of output is {}'.format(outputs.size()))
				log('The size of labels is {}'.format(labels.size()))
				loss = nn.L1Loss()
				labels = labels.type(torch.cuda.FloatTensor)
				outputs = torch.squeeze(outputs, dim=1)

				loss = loss(outputs, labels)
			else:
				loss = cross_entropy2d(outputs, labels)
			loss.backward()
			optimizer.step()
			loss_sum = loss_sum + torch.Tensor([loss.data]).unsqueeze(0).cpu()

		avg_loss = loss_sum / img_counter
		avg_loss_array = np.array(avg_loss)
		epoch_loss_array_total = np.concatenate((epoch_loss_array_total, [[avg_loss_array[0][0], epoch]]), axis=0)
		print('The current loss of epoch', epoch, 'is', avg_loss_array[0][0])
		# training model will be saved
		log('The variable avg_loss_array is {}'.format(avg_loss_array))
		writer.add_scalar('train_main_loss', avg_loss_array[0][0], epoch)

		if epoch % 1 == 0:
			torch.save(model, "runs/{}_{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch, rand_int))
	# I guess the shape is (epoch, 2)
	log('epoch_loss_array_total is {}'.format(epoch_loss_array_total))
	# The shape of epoch_loss_array_total is (epoch, 2)
	log('the shape of epoch_loss_array_total is {}'.format(epoch_loss_array_total.shape))
	epoch_loss_array_total = np.delete(arr=epoch_loss_array_total, obj=0, axis=0)
	log('the shape of epoch_loss_array_total after removal is {}'.format(epoch_loss_array_total.shape))
	loss_min_indice = np.argmin(epoch_loss_array_total, axis=0)
	log('The loss_min_indice is {}'.format(loss_min_indice))
	torch.save(model, "runs/{}_{}_{}_{}_{}_min.pkl".format(args.arch, args.dataset, args.feature_scale,
															loss_min_indice[0], rand_int))

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
