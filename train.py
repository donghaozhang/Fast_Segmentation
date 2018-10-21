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
from guotai_brats17.parse_config import parse_config
import random
from guotai_brats17.data_loader import DataLoader

DEBUG = False


def log(s):
	if DEBUG:
		print(s)


def train(args, guotai_loader):


	# Setup Dataloader
	print('###### Step One: Setup Dataloader')
	data_loader = get_loader(args.dataset)
	data_path = get_data_path(args.dataset)

	# For 2D dataset keep is_transform True
	# loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))

	# For 3D dataset keep is_transform False
	# loader = data_loader(data_path, is_transform=False, img_size=(args.img_rows, args.img_cols))
	if args.dataset == 'brats17_loader':
		loader = data_loader(guotai_loader)
	elif

	n_classes = 4
	trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

	# Setup Model
	print('###### Step Two: Setup Model')
	model = get_model(args.arch, n_classes)
	#model = torch.load('/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/bisenet3Dbrain_brats17_loader_1_251_3020_min.pkl')
	#model = torch.load('/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/2177/bisenet3Dbrain_brats17_loader_1_293_min.pkl')
	#model = torch.load('/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/9863/FCDenseNet57_brats17_loader_1_599.pkl')
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
	# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
			if args.arch == 'bisenet3Dbrain' or args.arch == 'unet3d_cls' or args.arch == 'FCDenseNet57':
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
			torch.save(model, "runs/{}/{}_{}_{}_{}.pkl".format(rand_int, args.arch, args.dataset, args.feature_scale, epoch))
	# I guess the shape is (epoch, 2)
	log('epoch_loss_array_total is {}'.format(epoch_loss_array_total))
	# The shape of epoch_loss_array_total is (epoch, 2)
	log('the shape of epoch_loss_array_total is {}'.format(epoch_loss_array_total.shape))
	epoch_loss_array_total = np.delete(arr=epoch_loss_array_total, obj=0, axis=0)
	log('the shape of epoch_loss_array_total after removal is {}'.format(epoch_loss_array_total.shape))
	loss_min_indice = np.argmin(epoch_loss_array_total, axis=0)
	log('The loss_min_indice is {}'.format(loss_min_indice))
	torch.save(model, "runs/{}/{}_{}_{}_{}_min.pkl".format(rand_int, args.arch, args.dataset, args.feature_scale,
															loss_min_indice[0]))
	sys.stdout = orig_stdout
	f.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Hyperparams')
	parser.add_argument('--arch', nargs='?', type=str, default='FCDenseNet57',
						help='Architecture to use [\'fcn8s, unet, segnet etc\']')
	parser.add_argument('--dataset', nargs='?', type=str, default='brats17_loader',
						help='Dataset to use [\'pascal, camvid, ade20k etc\']')
	parser.add_argument('--img_rows', nargs='?', type=int, default=256,
						help='Height of the input image')
	parser.add_argument('--img_cols', nargs='?', type=int, default=256,
						help='Height of the input image')
	parser.add_argument('--n_epoch', nargs='?', type=int, default=600,
						help='# of the epochs')
	parser.add_argument('--batch_size', nargs='?', type=int, default=1,
						help='Batch Size')
	parser.add_argument('--l_rate', nargs='?', type=float, default=1e-4,
						help='Learning Rate')
	parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
						help='Divider for # of features to use')
	parser.add_argument('--patch_size', nargs='?', type=int, default=[64, 64, 64],
						help='patch_size for training')
	parser.add_argument('--pretrained_path', nargs='?', type=str, default='empty',
						help='path for pretrained model')
	parser.add_argument('--n_classes', nargs='?', type=int, default=4,
						help='the number of class for classification')
	args = parser.parse_args()
	rand_int = np.random.randint(10000)

	# 1, load configuration parameters
	config_file_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/FCDenseNet57_train_87_wt_ax.txt'
	log('Load Configuration Parameters')
	config = parse_config(config_file_path)
	config_data = config['data']
	# print(config_data)
	config_net = config['network']
	config_train = config['training']
	random.seed(config_train.get('random_seed', 1))
	assert (config_data['with_ground_truth'])
	net_type = config_net['net_type']
	net_name = config_net['net_name']
	class_num = config_net['class_num']
	batch_size = config_data.get('batch_size', 5)

	# 2, construct graph
	log('Construct Graph')
	full_data_shape = [batch_size] + config_data['data_shape']
	full_label_shape = [batch_size] + config_data['label_shape']
	log('The full_label_shape is {}'.format(full_label_shape))
	log('The full_data_shape is {}'.format(full_data_shape))
	dataloader_guotai = DataLoader(config_data)
	dataloader_guotai.load_data()
	orig_stdout = sys.stdout
	writer = SummaryWriter('runs/' + str(rand_int))
	f = open("runs/{}/log.txt".format(rand_int), 'w')
	sys.stdout = f
	print('###### Step Zero: Log Number is ', rand_int)
	print('The dataset is {}'.format(args.dataset))
	print('The nettype is {}'.format(args.arch))
	print('The patch size is {}'.format(args.patch_size))
	print('The learning rate is {}'.format(args.l_rate))
	print('The total training epoch is {}'.format(args.n_epoch))
	print('The batch_size is {}'.format(args.batch_size))
	print('The pretrained path is {}'.format(args.pretrained_path))
	# print('pretrained: /home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/2177/bisenet3Dbrain_brats17_loader_1_293_min.pkl')
	train(args=args, guotai_loader=dataloader_guotai)
