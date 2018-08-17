import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import label
from PIL import Image as pil_image
cancer_types = ['gbm', 'hnsc', 'lgg', 'lung']
for type_num in range(0, 4):
    for pic_num in range(1, 9):
        program_name = 'python'
        test_script_path = ' test.py'
        model_args = ' --model_path'
        linknet_model_path = ' linknet_cellcancer_1_199.pkl'
        unet_model_path = ' unet_cellcancer_1_99.pkl'
        segnet_model_path = ' segnet_cellcancer_1_163.pkl'
        fcnnet_model_path = ' fcn8s_cellcancer_1_36.pkl'
        gcnnet_model_path = ' gcnnet_cellcancer_1_9.pkl'
        data_args = ' --dataset'
        data_type = ' cellcancer'
        img_path_args = ' --img_path '
        folder_num = 1000
        # type_num = 1
        prefix = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/maskrcnn'
        img_path = prefix + '/resize_test/image0' + str(pic_num) + '_' + cancer_types[type_num] + '_resized.png'
        # label_path = prefix + str(folder_num) \
        #              + '/test/' + str(pic_num)  + '_label.png'
        output_args = ' --out_path '
        u_img_save_path = prefix + '/compare_DH/' + str(pic_num) + '_' + cancer_types[type_num] + '_unet_resized.png'
        u_bi_img_save_path = prefix + '/compare_DH/' + str(pic_num) + '_' + cancer_types[type_num] + '_unet_bi_resized.png'
        link_img_save_path = prefix + '/compare_DH/' + str(pic_num) + '_' + cancer_types[type_num] + '_linknet_resized.png'
        link_bi_img_save_path = prefix + '/compare_DH/' + str(pic_num) + '_' + cancer_types[type_num] + '_linknet_bi_resized.png'
        gcn_img_save_path = prefix + '/compare_DH/' + str(pic_num) + '_' + cancer_types[type_num] + '_gcnnet_resized.png'
        gcn_bi_img_save_path = prefix + '/compare_DH/' + str(pic_num) + '_' + cancer_types[type_num] + '_gcnnet_bi_resized.png'
        seg_img_save_path = prefix + '/compare_DH/' + str(pic_num) + '_' + cancer_types[type_num] + '_segnet_resized.png'
        seg_bi_img_save_path = prefix + '/compare_DH/' + str(pic_num) + '_' + cancer_types[type_num] + '_segnet_bi_resized.png'
        # fcn_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
        #                     + '/test/' + str(pic_num) + '_resized_fcnnet_result.png'
        # ganunet_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/neural_network_learn/lib/pytorch-CycleGAN-and-pix2pix-master/results/pix2pixunet_v2/test_latest/images/' \
        #                         + str(pic_num) + '_fake_B.png'
        # ganunet_bi_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
        #                         + '/test/' + str(pic_num) + '_bi_ganunet_result.png'

        unet_pycmd = program_name + test_script_path + model_args + unet_model_path + data_args + data_type + img_path_args + img_path + output_args + u_img_save_path
        segnet_pycmd = program_name + test_script_path + model_args + segnet_model_path + data_args + data_type + img_path_args + img_path + output_args + seg_img_save_path
        linknet_pycmd = program_name + test_script_path + model_args + linknet_model_path + data_args + data_type + img_path_args + img_path + output_args + link_img_save_path
        gcnnet_pycmd = program_name + test_script_path + model_args + gcnnet_model_path + data_args + data_type + img_path_args + img_path + output_args + gcn_img_save_path
        # fcnnet_pycmd = program_name + test_script_path + model_args + fcnnet_model_path + data_args + data_type + img_path_args + img_path + output_args + fcn_img_save_path

        # os.system(unet_pycmd)
        # os.system(linknet_pycmd)
        # os.system(gcnnet_pycmd)
        # linknet_result = cv2.imread(link_img_save_path)
        linknet_result = cv2.imread(u_img_save_path)
        linknet_result = linknet_result[:,:,0]
        linknet_result = linknet_result > 254
        linknet_result = linknet_result.astype('uint8')
        linknet_result = linknet_result * 200
        # linknet_result = label(linknet_result, neighbors=4, connectivity=1)
        print('the shape of linknet', linknet_result.shape)
        linknet_result = pil_image.fromarray(linknet_result)
        # linknet_result.save(link_bi_img_save_path)
        linknet_result.save(u_bi_img_save_path)

        # fig = plt.figure()
        # plt.imshow(linknet_result)
        # plt.show()
        # os.system(segnet_pycmd)
        # os.system(fcnnet_pycmd)