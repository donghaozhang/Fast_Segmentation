import os
import cv2
import numpy as np
from PIL import Image as pil_image
from skimage.transform import resize
import matplotlib.pyplot as plt
cancer_type = ['gbm', 'hnsc', 'lgg', 'lung']
def processimg(imgpath):
    im_cv2 = cv2.imread(imgpath)
    im_cv2 = np.flip(im_cv2, 2)
    return im_cv2
for det in cancer_type:
    for i in range(1, 9):
        program_name = 'python'
        test_script_path = ' test.py'
        model_args = ' --model_path'
        model_path = ' unet_cellcancer_1_49.pkl'
        data_args = ' --dataset'
        data_type = ' cellcancer'
        img_path_args = ' --img_path '
        img_path = '/home/neuron/Desktop/Donghao/cellsegmentation/segmentation-test-images/test_original/' + det + '/test-images/image0' + str(i) + '.png'
        output_args = ' --out_path '
        result_path = '/home/neuron/Desktop/Donghao/cellsegmentation/tmp_testing/result_' + det + '_' + str(i) + '.png'
        pycmd = program_name + test_script_path + model_args + model_path + data_args + data_type + img_path_args + img_path + output_args + result_path
        cur_test_img = processimg(img_path)
        if cur_test_img.shape[0] == 500:
            cur_test_img_resize = resize(cur_test_img, (512, 512, 3), mode='reflect')
            cur_test_img_resize = cur_test_img_resize * 255
            cur_test_img_resize = cur_test_img_resize.astype('uint8')
            cur_test_img_resize_save = pil_image.fromarray(cur_test_img_resize)
            cur_test_img_resize_save.save(result_path)

        elif cur_test_img.shape[0] == 600:
            cur_test_img_resize = resize(cur_test_img, (640, 640, 3), mode='reflect')
            cur_test_img_resize = cur_test_img_resize * 255
            cur_test_img_resize = cur_test_img_resize.astype('uint8')
            cur_test_img_resize_save = pil_image.fromarray(cur_test_img_resize)
            cur_test_img_resize_save.save(result_path)
