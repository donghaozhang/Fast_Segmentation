from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pretrainedmodels
from ptsemseg.models.xceptionnet import Xception

im_filepath = "/home/donghao/Desktop/donghao/fast_segmentation/digital_pathology_2018/Digital Pathology_segmentation_training_set/train/03.png"
# im_filepath = "/home/donghao/Desktop/donghao/fast_segmentation/digital_pathology_2018/Digital Pathology_segmentation_training_set/train/03.png"
# load from original image to figure mysterious 4 matrix
# original_image_path = '/home/donghao/Desktop/donghao/fast_segmentation/digital_pathology_2018/Digital Pathology_segmentation_training_set/image01.png'
original_image_path = im_filepath
im_np = cv2.imread(original_image_path)
im_np = np.asarray(im_np)
im_np = im_np.astype(float)
print(im_np[:, :, 2])
print(im_np.dtype)
print(im_np.shape)
# im_np_show = im_np / 255
# print(im_np)
# im = Image.open(original_image_path)
# im_np = np.asarray(im)
# print(im_np.shape)
# print(im_np[:,:,2])
fig1 = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(im_np[:, :, 0])

plt.subplot(2, 2, 2)
plt.imshow(im_np[:, :, 1])

plt.subplot(2, 2, 3)
plt.imshow(im_np[:, :, 2])

print(pretrainedmodels.model_names)
# model_name = 'nasnetalarge' # could be fbresnet152 or inceptionresnetv2
# model_name = 'xception'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# print(model)
xxxx = Xception()
# model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# print(model)
# model.eval()

# plt.imshow(im_np[:,:,3])0

# plt.subplot(2, 2, 4)
# plt.show()