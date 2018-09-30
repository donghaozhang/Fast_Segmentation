from ptsemseg.models.xception39 import xception39
from ptsemseg.models.xception39 import bisenet
from ptsemseg.models.xception39 import bisenet3D
import numpy as np
import torch
from torch.autograd import Variable


# bisenet_model = bisenet(num_classes=1000, pretrained=False)
# bisenet_model.cuda()
# xception_model = xception39(num_classes=1000, pretrained=False)
# xception_model.cuda()

# 2D version of numpy_fake_image should be your 3D input block
# fake_im_num = 1
# numpy_fake_image = np.random.rand(fake_im_num, 3, 224, 224)
# tensor_fake_image = torch.FloatTensor(numpy_fake_image)
# torch_fake_image = Variable(tensor_fake_image).cuda()
# output = bisenet_model(torch_fake_image)
# print('the output size is : ', output.size())

from scipy.stats import norm
# a = [26/8, 32/8, 34/6, 28/6]
# mean, std = np.mean(a), np.std(a)
# b = a/np.sum(a)
# print(300*b)

# Try to create 3D fake data
fake_im_num = 1
bisenet_model_3D = bisenet3D(num_classes=1000, pretrained=False)
bisenet_model_3D.cuda()
numpy_fake_image_3d = np.random.rand(fake_im_num, 3, 224, 224, 224)
tensor_fake_image_3d = torch.FloatTensor(numpy_fake_image_3d)
torch_fake_image_3d = Variable(tensor_fake_image_3d).cuda()
output_3d = bisenet_model_3D(torch_fake_image_3d)
