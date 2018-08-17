from os import path
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import cv2

# Load the txt file storing the mask information
absFilePath = os.path.abspath(__file__)
folder_name = os.path.dirname(absFilePath)
prefix_txt = "fast_segmentation/digital_pathology_2018/Digital Pathology_segmentation_training_set/image"
file_num = 1
file_num_txt = str(0) + str(file_num)
suffix_txt = "_mask.txt"
im_suffix_txt = ".png"
txt_comb = prefix_txt + file_num_txt + suffix_txt
im_comb = prefix_txt + file_num_txt + im_suffix_txt
txt_filepath = path.join(folder_name, "..", txt_comb)
im_filepath = path.join(folder_name, "..", im_comb)

text_file = open(txt_filepath, "r")

# Divide text_file into multiple lines so that I can easily index them
lines = text_file.readlines()

# String format of the image dimension
txt_im_dim = lines[0]
# print(lines[1])
txt_im_dim_split = txt_im_dim.split()
width_num = int(txt_im_dim_split[0])
height_num = int(txt_im_dim_split[1])
print('the image dimension of the current file is :', txt_im_dim_split[0])
print('the length of lines', len(lines))
print('width_num and height_num are : ', width_num, height_num)

label_im = np.zeros((width_num, height_num))
im = Image.open(im_filepath)

for x in range(0, width_num):
	for y in range(0, height_num):
		loc_num = x + y * width_num + 1
		label_im[x, y] = int(lines[loc_num])

print('the shape of label_im is ', label_im.shape)
label_im_transpose = np.transpose(label_im, (1, 0))
print('the shape of label_im_transpose is ', label_im_transpose.shape)
print('the final loc_num is ', loc_num)
fig1 = plt.figure()
plt.imshow(label_im_transpose)
fig2 = plt.figure()
plt.imshow(im)
plt.show()
