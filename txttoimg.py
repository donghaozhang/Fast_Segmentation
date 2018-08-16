from os import path
import os
# Load the txt file storing the mask information
absFilePath = os.path.abspath(__file__)
folder_name = os.path.dirname(absFilePath)
filepath = path.join(folder_name, "..", "fast_segmentation/digital_pathology_2018/Digital Pathology_segmentation_training_set/image01_mask.txt")
text_file = open(filepath, "r")

# Divide text_file into multiple lines so that I can easily index them
lines = text_file.readlines()

# String format of the image dimension
txt_im_dim = lines[0]
txt_im_dim_split = txt_im_dim.split()
print('the image dimension of the current file is :', txt_im_dim_split[0])
print('the length of lines', len(lines))
