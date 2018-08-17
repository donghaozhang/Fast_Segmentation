import os
import cv2
import numpy as np
from PIL import Image as pil_image
# os.system('python train.py --arch fcn32s --dataset cellcancer --n_epoch 1 --batch_size 4')
# os.system('python train.py --arch fcn32s --dataset cellcancer --n_epoch 1 --batch_size 4')

def processimg(imgpath):
    im_cv2 = cv2.imread(imgpath)
    im_cv2 = np.flip(im_cv2, 2)
    return im_cv2
cancer_type = ['gbm', 'hnsc', 'lgg', 'lung']
counter = 0

# # testing for cross validation
# for det in cancer_type:
#     for i in range(1, 9):
#         counter = counter + 1
#         program_name = 'python'
#         test_script_path = ' test.py'
#         model_args = ' --model_path'
#         model_path = ' unet_cellcancer_1_49.pkl'
#         data_args = ' --dataset'
#         data_type = ' cellcancer'
#         img_path_args = ' --img_path '
#         img_path = '/home/neuron/Desktop/Donghao/cell_data/segmentation-train-images/' + det + '/training-set/image0' + str(i) + '.png'
#         label_path = '/home/neuron/Desktop/Donghao/cell_data/segmentation-train-images/' + det + '/training-set/image0' + str(i) + '_mask.png'
#         boundary_path = '/home/neuron/Desktop/Donghao/cell_data/segmentation-train-images/' + det + '/training-set/image0' + str(i) + '_poly.png'
#         output_args = ' --out_path '
#         img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(i) + '/test/' + str(i) + '_' + det + '.png'
#         label_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(i) + '/test/' + str(i) + '_' + det +'_label' + '.png'
#         boundary_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(i) + '/test/' + str(i) + '_' + det + '_boundary' + '.png'
#         # pycmd = program_name + test_script_path + model_args + model_path + data_args + data_type + img_path_args + img_path + output_args + result_path
#         cur_test_img = processimg(img_path)
#         cur_test_img = cur_test_img.astype('uint8')
#         cur_test_img_save = pil_image.fromarray(cur_test_img)
#         cur_test_img_save.save(img_save_path)
#         cur_test_label = processimg(label_path)
#         cur_test_label = np.asarray(cur_test_label)
#         cur_test_label = cur_test_label.astype('uint8')
#         cur_test_label_save = pil_image.fromarray(cur_test_label)
#         cur_test_label_save.save(label_save_path)
#         cur_test_boundary = processimg(boundary_path)
#         # print(cur_test_boundary.shape)
#         cur_test_boundary = cur_test_boundary.astype('uint8')
#         cur_test_boundary_save = pil_image.fromarray(cur_test_boundary)
#         cur_test_boundary_save.save(boundary_save_path)
#

odds = [i for i in range(9) if i%2==1]
evens = [i for i in range(1,9) if i%2==0]
print(odds, evens)
# training for cross validation
counter = 0
for det in cancer_type:
        for pic_num in odds:
                counter = counter + 1
                program_name = 'python'
                test_script_path = ' test.py'
                model_args = ' --model_path'
                model_path = ' unet_cellcancer_1_49.pkl'
                data_args = ' --dataset'
                data_type = ' cellcancer'
                img_path_args = ' --img_path '
                img_path = '/home/neuron/Desktop/Donghao/cell_data/segmentation-train-images/' + det + '/training-set/image0' + str(pic_num) + '.png'
                label_path = '/home/neuron/Desktop/Donghao/cell_data/segmentation-train-images/' + det + '/training-set/image0' + str(pic_num) + '_mask.png'
                boundary_path = '/home/neuron/Desktop/Donghao/cell_data/segmentation-train-images/' + det + '/training-set/image0' + str(pic_num) + '_poly.png'
                output_args = ' --out_path '
                # img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val1/train/' + str(counter) + '.png'
                # label_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val1/train/' + str(counter) + '_label' + '.png'
                # boundary_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val1/train/' + str(counter) + '_boundary' + '.png'
                img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val2/test/' + str(counter) + '.png'
                label_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val2/test/' + str(counter) + '_label' + '.png'
                boundary_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val2/test/' + str(counter) + '_boundary' + '.png'
                # pycmd = program_name + test_script_path + model_args + model_path + data_args + data_type + img_path_args + img_path + output_args + result_path

                cur_test_img = processimg(img_path)
                cur_test_img = cur_test_img.astype('uint8')
                cur_test_img_save = pil_image.fromarray(cur_test_img)
                cur_test_img_save.save(img_save_path)

                # object number labelling
                cur_test_label = processimg(label_path)
                cur_test_label = np.asarray(cur_test_label)
                cur_test_label = cur_test_label.astype('uint8')
                cur_test_label_save = pil_image.fromarray(cur_test_label)
                cur_test_label_save.save(label_save_path)

                # cell with yellow boundary
                cur_test_boundary = processimg(boundary_path)
                cur_test_boundary = cur_test_boundary.astype('uint8')
                cur_test_boundary_save = pil_image.fromarray(cur_test_boundary)
                cur_test_boundary_save.save(boundary_save_path)

# path_prefix = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val'
# for i in range(1, 9):
#     os.makedirs(path_prefix + str(i) + '/comp_exp/train')
#     os.makedirs(path_prefix + str(i) + '/comp_exp/test')
#     os.makedirs(path_prefix + str(i) + '/comp_exp/val')
#     os.makedirs(path_prefix + str(i) + '/comp_exp/trainannot')
#     os.makedirs(path_prefix + str(i) + '/comp_exp/testannot')
#     os.makedirs(path_prefix + str(i) + '/comp_exp/valannot')

# for x in range(1, 9):
#     os.makedirs('val'+str(x)+'/train')
#     os.makedirs('val'+str(x)+'/test')


# path_prefix = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val'
# folder_num = 1
# for det in cancer_type:
#         counter = counter + 1
#         program_name = 'python'
#         test_script_path = ' test.py'
#         model_args = ' --model_path'
#         linknet_model_path = ' linknet_cellcancer_1_175.pkl'
#         unet_model_path = ' /home/neuron/Desktop/Donghao/cellsegmentation/trained_model_history/unet_cellcancer_1_49.pkl'
#         segnet_model_path = ' segnet_cellcancer_1_80.pkl'
#         data_args = ' --dataset'
#         data_type = ' cellcancer'
#         img_path_args = ' --img_path '
#         img_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) + '/test/' + str(folder_num) + '_' + str(det) + '_resized.png'
#         label_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) + '/test/' + str(folder_num) + '_' + str(det) + '_label.png'
#         boundary_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) + '/test/' + str(folder_num) + '_' + str(det) + '_poly.png'
#         output_args = ' --out_path '
#         u_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) + '/test/' + str(folder_num) + '_' + str(det) + '_resized_unet_result.png'
#         link_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) + '/test/' + str(folder_num) + '_' + str(det) + '_resized_linknet_result.png'
#         seg_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) + '/test/' + str(folder_num) + '_' + str(det) + '_resized_segnet_result.png'
#
#         label_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) + '/test/' + str(folder_num) + '_' + str(det) + '_label_result.png'
#         boundary_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) + '/test/' + str(folder_num) + '_' + str(det) + '_poly_result.png'
#         unet_pycmd = program_name + test_script_path + model_args + unet_model_path + data_args + data_type + img_path_args + img_path + output_args + u_img_save_path
#         segnet_pycmd = program_name + test_script_path + model_args + segnet_model_path + data_args + data_type + img_path_args + img_path + output_args + seg_img_save_path
#         linknet_pycmd = program_name + test_script_path + model_args + linknet_model_path + data_args + data_type + img_path_args + img_path + output_args + link_img_save_path
#
#         os.system(unet_pycmd)
#         os.system(linknet_pycmd)
#         os.system(segnet_pycmd)