for i in range(3):
	print(i)
from ptsemseg.models.xceptionnet import xception
import numpy as np
import torch
from torch.autograd import Variable
xception_model = xception(num_classes=1000, pretrained=False)
xception_model.cuda()

# numpy_fake_image should be your 3D input block

numpy_fake_image = np.random.rand(20, 3, 224, 224)
tensor_fake_image = torch.FloatTensor(numpy_fake_image)
torch_fake_image = Variable(tensor_fake_image).cuda()
output = xception_model(torch_fake_image)
print(output.size())