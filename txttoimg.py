import numpy as np
text_file = open("./Digital Pathology_segmentation_training_set/image01_mask.txt", "r")
lines = text_file.readlines()
print(lines[0])
im_dim = lines[0]
# x, y = np.loadtxt(im_dim, delimiter=',', usecols=(0, 2), unpack=True)
test_split = im_dim.split()
print(test_split)
print(im_dim[0])
print('the length of lines', len(lines))
