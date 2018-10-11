import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.segnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.linknet import *
from ptsemseg.models.gcnnet import *
from ptsemseg.models.bisenet3Dbrain import Bisenet3DBrain
from ptsemseg.models.unet3d_brain import unet3d

# from ptsemseg.models.bisenet3D import *


def get_model(name, n_classes):
	model = _get_model_instance(name)

	if name in ['fcn32s', 'fcn16s', 'fcn8s']:
		model = model(n_classes=n_classes)
		# vgg16 = models.vgg16(pretrained=True)
		# model.init_vgg16_params(vgg16)

	elif name == 'segnet':
		model = model(n_classes=n_classes,
					  is_unpooling=True)
		# vgg16 = models.vgg16(pretrained=True)
		# model.init_vgg16_params(vgg16)

	elif name == 'unet':
		model = model(n_classes=n_classes,
					  is_batchnorm=True,
					  in_channels=3,
					  is_deconv=True)

	elif name == 'linknet':
		model = model(n_classes=n_classes,
					  is_batchnorm=True,
					  in_channels=3,
					  is_deconv=True)

	elif name == 'pspnet':
		model = model(n_classes=n_classes,
					  is_batchnorm=True,
					  in_channels=3,
					  is_deconv=True)

	elif name == 'gcnnet':
		model = model(num_classes=n_classes,
					  input_size=256,
					  pretrained=True)
	# elif name == 'bisenet3D':
	#	model = model()

	elif name == 'bisenet3Dbrain':
		model = model()

	elif name == 'unet3d_res':
		model = model(feature_scale=4, n_classes=1, is_deconv=True, in_channels=4)

	elif name == 'unet3d_cls':
		model = model(feature_scale=4, n_classes=4, is_deconv=True, in_channels=4)

	else:
		raise 'Model {} not available'.format(name)

	return model


def _get_model_instance(name):
	return {
		'fcn32s': fcn32s,
		'fcn8s': fcn8s,
		'fcn16s': fcn16s,
		'unet': unet,
		'segnet': segnet,
		'pspnet': pspnet,
		'linknet': linknet,
		'gcnnet': GCN,
		'bisenet3Dbrain': Bisenet3DBrain,
		'unet3d_res': unet3d,
		'unet3d_cls': unet3d,
		# 'bisenet3D': Bisenet3D,
	}[name]
