import os
import cv2
import numpy as np
from PIL import Image as pil_image
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.morphology import label
import csv
from multiprocessing import Process
from multiprocessing import Pool
# os.system('python train.py --arch fcn32s --dataset cellcancer --n_epoch 1 --batch_size 4')
# os.system('python train.py --arch fcn32s --dataset cellcancer --n_epoch 1 --batch_size 4')

#An example of a class
class Method_Region:

    def __init__(self):
        self.label_num_list = np.asarray([])
        self.precision_list = np.asarray([])
        self.recall_list = np.asarray([])
        self.fscore_list = np.asarray([])

    def add_label_num(self, label_num):
        self.label_num_list = np.append(self.label_num_list, label_num)
        # self.label_num_list = self.label_num_list.append(label_num)

    def add_precision(self, precision_val):
        #self.precision_list = self.precision_list.append(precision_val)
        self.precision_list = np.append(self.precision_list, precision_val)

    def add_recall(self, recall_val):
        self.recall_list = np.append(self.recall_list, recall_val)
        #self.recall_list = self.recall_list.append(recall_val)

    # def add_fscore(self, fscore_val):
    #     self.fscore_list = np.append(self.fscore_list, fscore_val)
        # self.fscore_list = self.fscore_list.append(fscore_val)

def processimg(imgpath):
    im_cv2 = cv2.imread(imgpath)
    im_cv2 = np.flip(im_cv2, 2)
    return im_cv2
cancer_type = ['gbm', 'hnsc', 'lgg', 'lung']
counter = 0

def accurate(im, gt):
    """ im is the prediction result;
        gt is the ground truth labelled by biologists;"""
    tp = np.logical_and(im, gt)
    tp = tp.sum()
    precision = tp.sum() / im.sum()
    recall = tp.sum() / gt.sum()
    if tp > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0
    return precision, recall, fscore

def tp_pixel_num_cal(im, gt):
    """ im is the prediction result;
        gt is the ground truth labelled by biologists;"""
    tp = np.logical_and(im, gt)
    tp_pixel_num = tp.sum()
    return tp_pixel_num

def evaluate_one_pic(pic_num):
    path_prefix = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val'
    folder_num = 2
    fscore_link_sum = 0
    fscore_ganunet_sum = 0
    fscore_counet_sum = 0
# for pic_num in range(1, 17):
#     counter = counter + 1
    program_name = 'python'
    test_script_path = ' test.py'
    model_args = ' --model_path'
    linknet_model_path = ' linknet_cellcancer_1_380.pkl'
    unet_model_path = ' unet_cellcancer_1_150.pkl'
    segnet_model_path = ' segnet_cellcancer_1_163.pkl'
    fcnnet_model_path = ' fcn8s_cellcancer_1_36.pkl'
    data_args = ' --dataset'
    data_type = ' cellcancer'
    img_path_args = ' --img_path '
    img_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
               + '/test/' + str(pic_num)  + '_resized.png'
    label_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
                 + '/test/' + str(pic_num)  + '_label.png'
    boundary_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) \
                    + '/test/' + str(pic_num) + '_poly.png'
    output_args = ' --out_path '
    u_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
                      + '/test/' + str(pic_num)  + '_resized_unet_result.png'
    u_bi_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
                      + '/test/' + str(pic_num)  + '_bi_unet_result.png'
    link_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
                         + '/test/' + str(pic_num) + '_resized_linknet_result.png'
    link_bi_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
                            + '/test/' + str(pic_num) + '_bi_linknet_result.png'
    seg_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
                        + '/test/' + str(pic_num) + '_resized_segnet_result.png'
    seg_bi_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
                        + '/test/' + str(pic_num) + '_bi_segnet_result.png'
    fcn_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
                        + '/test/' + str(pic_num) + '_resized_fcnnet_result.png'
    ganunet_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/neural_network_learn/lib/pytorch-CycleGAN-and-pix2pix-master/results/pix2pixunet_v2_part2/test_latest/images/' \
                            + str(pic_num) + '_fake_B.png'
    ganunet_bi_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
                            + '/test/' + str(pic_num) + '_bi_ganunet_result.png'
    ganresnet_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/neural_network_learn/lib/pytorch-CycleGAN-and-pix2pix-master/results/pix2pixresnet_v2_part2/test_latest/images/' \
                            + str(pic_num) + '_fake_B.png'
    ganresnet_bi_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val' + str(folder_num) \
                            + '/test/' + str(pic_num) + '_bi_ganresnet_result.png'
    # counet_img_save_path = '//home/neuron/Desktop/Donghao/cellsegmentation/neural_network_learn/lib/pytorch-CycleGAN-and-pix2pix-master/results/multitask/test_latest/images/1_'\
    #                        + str(det) + '_fake_B.png'
    # counet_bi_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) \
    #                         + '/test/' + str(folder_num) + '_' + str(det) + '_bi_counet_result.png'
    counet_label_img_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/neural_network_learn/lib/pytorch-CycleGAN-and-pix2pix-master/results/multitask_v2_part2/test_latest' \
                                 + '/images' + str(pic_num)  + '_label_object_final.png'
    # label_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) + '/test/' + str(folder_num) + '_' + str(det) + '_label.png'
    # boundary_save_path = '/home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val' + str(folder_num) + '/test/' + str(folder_num) + '_' + str(det) + '_poly_result.png'
    unet_pycmd = program_name + test_script_path + model_args + unet_model_path + data_args + data_type + img_path_args + img_path + output_args + u_img_save_path
    segnet_pycmd = program_name + test_script_path + model_args + segnet_model_path + data_args + data_type + img_path_args + img_path + output_args + seg_img_save_path
    linknet_pycmd = program_name + test_script_path + model_args + linknet_model_path + data_args + data_type + img_path_args + img_path + output_args + link_img_save_path
    fcnnet_pycmd = program_name + test_script_path + model_args + fcnnet_model_path + data_args + data_type + img_path_args + img_path + output_args + fcn_img_save_path
    # os.system(unet_pycmd)
    # os.system(linknet_pycmd)
    # os.system(segnet_pycmd)
    # os.system(fcnnet_pycmd)

    # 0.93 linknet 0.9 unet 0.7 ganresnet
    # cur_img = processimg(seg_img_save_path)
    # cur_img = processimg(link_img_save_path)
    # cur_img = processimg(counet_img_save_path)
    # cur_img = processimg(u_img_save_path)
    # cur_img = processimg(ganunet_img_save_path)
    # cur_img = processimg(ganresnet_img_save_path)
    # if cur_img.shape[0] == 640:
    #         cur_img = resize(cur_img, (600, 600, 3), mode='reflect')
    # elif cur_img.shape[0] == 1280:
    #         cur_img = resize(cur_img, (600, 600, 3), mode='reflect')
    # elif cur_img.shape[1] == 512:
    #         cur_img = resize(cur_img, (500, 500, 3), mode='reflect')
    # fig = plt.figure()
    # plt.imshow(cur_img)
    # plt.title("cur_img_ganresnet_label")
    # plt.show()
    # bi_cur_img = cur_img[:, :, 0] > 0.5
    # bi_cur_img = bi_cur_img * 255
    # bi_cur_img = bi_cur_img.astype('uint8')
    # bi_cur_img = pil_image.fromarray(bi_cur_img)
    # bi_cur_img.save(counet_bi_img_save_path)
    # bi_cur_img.save(ganunet_bi_img_save_path)
    # bi_cur_img.save(link_bi_img_save_path)
    # bi_cur_img.save(seg_bi_img_save_path)
    # bi_cur_img.save(u_bi_img_save_path)
    # bi_cur_img.save(ganresnet_bi_img_save_path)

    # fig = plt.figure()
    # plt.imshow(bi_cur_img)
    # plt.title("linknet image")
    # plt.show()
    #
    # cur_img_link = processimg(link_bi_img_save_path)
    # cur_img_link = cur_img_link[:,:,0] > 10
    # cur_img_link_label = label(cur_img_link, neighbors=4, connectivity=1)
    #
    # cur_img_ganunet = processimg(ganunet_bi_img_save_path)
    # cur_img_ganunet = cur_img_ganunet[:,:,0] > 10
    # cur_img_ganunet_label = label(cur_img_ganunet, neighbors=4, connectivity=1)
    # cur_img_ganresnet = processimg(ganresnet_bi_img_save_path)
    # cur_img_ganresnet = cur_img_ganresnet[:,:,0] > 10
    # cur_img_ganresnet_label = label(cur_img_ganresnet, neighbors=4, connectivity=1)
    #
    # cur_img_unet = processimg(u_bi_img_save_path)
    # cur_img_unet = cur_img_unet[:,:,0] > 10
    # cur_img_unet_label = label(cur_img_unet, neighbors=4, connectivity=1)
    #
    # cur_img_segnet = processimg(seg_bi_img_save_path)
    # cur_img_segnet = cur_img_segnet[:,:,0] > 10
    # cur_img_segnet_label = label(cur_img_segnet, neighbors=4, connectivity=1)
    # fig = plt.figure()
    # plt.imshow(cur_img_ganresnet)
    # plt.title("cur_img_ganresnet_label")
    # plt.show()
    # #
    # # cur_img_counet = processimg(counet_bi_img_save_path)
    # # cur_img_counet = cur_img_counet[:,:,0] > 10
    # # # cur_img_counet_label = label(cur_img_counet, neighbors=4, connectivity=1)
    cur_img_counet_label_img = processimg(counet_label_img_save_path)
    cur_img_counet_label = cur_img_counet_label_img[:,:,0]
    # cur_img_counet = cur_img_counet_label > 0
    # fig = plt.figure()
    # plt.imshow(cur_img_counet_label)
    # plt.title("cur_img_label_single")
    # plt.show()
    # #
    # #
    cur_img_label = processimg(label_path)
    cur_img_label_single = cur_img_label[:,:,0]
    cur_img_label_single_bi = cur_img_label_single > 0
    # fig = plt.figure()
    # plt.imshow(cur_img_label_single)
    # plt.title("cur_img_label_single")
    # plt.show()
    cur_img_label_gt_region = np.zeros(cur_img_label_single.shape)
    cur_img_label_link_region = np.zeros(cur_img_label_single.shape)
    cur_img_label_ganunet_region = np.zeros(cur_img_label_single.shape)
    cur_img_label_segnet_region = np.zeros(cur_img_label_single.shape)
    cur_img_label_unet_region = np.zeros(cur_img_label_single.shape)
    cur_img_label_counet_region = np.zeros(cur_img_label_single.shape)
    cur_img_label_ganresnet_region = np.zeros(cur_img_label_single.shape)
    link_counter = 0
    ganunet_counter = 0
    ganresnet_counter = 0
    counet_counter = 0
    unet_counter = 0
    segnet_counter = 0
    tp_pixel_num_sum_counet = 0
    tp_pixel_num_sum_link = 0
    tp_pixel_num_sum_ganunet = 0
    for label_num_gt in range(1, cur_img_label_single.max() + 1):
        cur_img_label_gt_region[:, :] = 0
        cur_img_label_gt_region[cur_img_label_single == label_num_gt] = True
        # for label_num_link in range(1, cur_img_link_label.max() + 1):
        #     cur_img_label_link_region[:, :] = 0
        #     cur_img_label_link_region[cur_img_link_label == label_num_link] = True
    #         precision_link, recall_link, fscore_link = accurate(cur_img_label_link_region, cur_img_label_gt_region)
    #         if (precision_link > 0.5) & (recall_link > 0.5):
    #             link_counter = link_counter + 1
    #             # tp_pixel_num_link = tp_pixel_num_cal(cur_img_label_link_region, cur_img_label_gt_region)
    #
        # for label_num_ganunet in range(1, cur_img_ganunet_label.max() + 1):
        #     cur_img_label_ganunet_region[:, :] = 0
        #     cur_img_label_ganunet_region[cur_img_ganunet_label == label_num_ganunet] = True
        #     precision_ganunet, recall_ganunet, fscore_ganunet = accurate(cur_img_label_ganunet_region, cur_img_label_gt_region)
        #     if ((precision_ganunet > 0.5) & (recall_ganunet > 0.5)):
        #         ganunet_counter = ganunet_counter + 1
    #             # tp_pixel_num_ganunet = tp_pixel_num_cal(cur_img_label_ganunet_region, cur_img_label_gt_region)
    #
    #     for label_num_unet in range(1, cur_img_unet_label.max() + 1):
    #         cur_img_label_unet_region[:, :] = 0
    #         cur_img_label_unet_region[cur_img_unet_label == label_num_unet] = True
    #         precision_unet, recall_unet, fscore_unet = accurate(cur_img_label_unet_region,
    #                                                                      cur_img_label_gt_region)
    #         if ((precision_unet > 0.5) & (recall_unet > 0.5)):
    #             unet_counter = unet_counter + 1
    #             # tp_pixel_num_ganunet = tp_pixel_num_cal(cur_img_label_ganunet_region,
    #             #                                         cur_img_label_gt_region)
    #             # tp_pixel_num_sum_ganunet = tp_pixel_num_sum_ganunet + tp_pixel_num_ganunet
    #
        # for label_num_segnet in range(1, cur_img_segnet_label.max() + 1):
        #     cur_img_label_segnet_region[:, :] = 0
        #     cur_img_label_segnet_region[cur_img_segnet_label == label_num_segnet] = True
        #     precision_segnet, recall_segnet, fscore_segnet = accurate(cur_img_label_segnet_region,
        #                                                               cur_img_label_gt_region)
        #     if ((precision_segnet > 0.5) & (recall_segnet > 0.5)):
        #         segnet_counter = segnet_counter + 1

        for label_num_counet in range(1, cur_img_counet_label.max() + 1):
            cur_img_label_counet_region[:, :] = 0
            cur_img_label_counet_region[cur_img_counet_label == label_num_counet] = True
            precision_counet, recall_counet, fscore_counet = accurate(cur_img_label_counet_region, cur_img_label_gt_region)
            if ((precision_counet > 0.5) & (recall_counet > 0.5)):
                counet_counter = counet_counter + 1
        # for label_num_ganresnet in range(1, cur_img_ganresnet_label.max() + 1):
        #     cur_img_label_ganresnet_region[:, :] = 0
        #     cur_img_label_ganresnet_region[cur_img_ganresnet_label == label_num_ganresnet] = True
        #     precision_ganresnet, recall_ganresnet, fscore_ganresnet = accurate(cur_img_label_ganresnet_region,
        #                                                               cur_img_label_gt_region)
        #     if ((precision_ganresnet > 0.5) & (recall_ganresnet > 0.5)):
        #         ganresnet_counter = ganresnet_counter + 1
                # tp_pixel_num_counet = tp_pixel_num_cal(cur_img_label_counet_region, cur_img_label_gt_region)
                # tp_pixel_num_sum_counet = tp_pixel_num_sum_counet + tp_pixel_num_counet
    #
    # cur_link_precision = link_counter / cur_img_link_label.max()
    # cur_link_recall = link_counter / cur_img_label_single.max()
    # cur_link_fscore = 2 * cur_link_precision * cur_link_recall / (cur_link_precision + cur_link_recall)
    # # cur_link_dice = 2 * link_counter / (cur_img_link_label.max() + cur_img_label_single.max())
    #
    #
    # cur_ganunet_precision = ganunet_counter / cur_img_ganunet_label.max()
    # cur_ganunet_recall = ganunet_counter / cur_img_label_single.max()
    # cur_ganunet_fscore = 2 * cur_ganunet_precision * cur_ganunet_recall / (cur_ganunet_precision + cur_ganunet_recall)
    # cur_ganresnet_precision = ganresnet_counter / cur_img_ganresnet_label.max()
    # cur_ganresnet_recall = ganresnet_counter / cur_img_label_single.max()
    # cur_ganresnet_fscore = 2 * cur_ganresnet_precision * cur_ganresnet_recall / (cur_ganresnet_precision + cur_ganresnet_recall)

    # cur_unet_precision = unet_counter / cur_img_unet_label.max()
    # cur_unet_recall = unet_counter / cur_img_label_single.max()
    # cur_unet_fscore = 2 * cur_unet_precision * cur_unet_recall / (cur_unet_precision + cur_unet_recall)
    #
    # cur_segnet_precision = segnet_counter / cur_img_segnet_label.max()
    # cur_segnet_recall = segnet_counter / cur_img_label_single.max()
    # cur_segnet_fscore = 2 * cur_segnet_precision * cur_segnet_recall / (cur_segnet_precision + cur_segnet_recall)
    # cur_ganunet_dice = 2 * ganunet_counter / (cur_img_ganunet_label.max() + cur_img_label_single.max())

    cur_counet_precision = counet_counter / cur_img_counet_label.max()
    cur_counet_recall = counet_counter / cur_img_label_single.max()
    cur_counet_fscore = 2 * cur_counet_precision * cur_counet_recall / (cur_counet_precision + cur_counet_recall)
    # cur_counet_dice = 2 * counet_counter / (cur_img_counet_label.max() + cur_img_label_single.max())
    # # print(cur_link_dice, cur_ganunet_dice, cur_counet_dice)
    # print(cur_segnet_precision, cur_segnet_recall, cur_segnet_fscore, cur_unet_precision, cur_unet_recall, cur_unet_fscore,
    #       cur_link_precision, cur_link_recall, cur_link_fscore, cur_ganunet_precision, cur_ganunet_recall, cur_ganunet_fscore)
    # print(cur_link_precision, cur_link_recall, cur_link_fscore)
    # print('the picture being tested is ', pic_num)
    print(cur_counet_precision, cur_counet_recall, cur_counet_fscore)
    # print(cur_ganresnet_precision, cur_ganresnet_recall, cur_ganresnet_fscore)
    # print(cur_ganunet_precision, cur_ganunet_recall, cur_ganunet_fscore)
    # print(cur_segnet_precision, cur_segnet_recall, cur_segnet_fscore)
    # #print(cur_link_recall, cur_ganunet_recall, cur_counet_recall)
    #print(cur_link_fscore, cur_ganunet_fscore, cur_counet_fscore)
    # # with open('metrics.csv', 'w', newline='') as csvfile:
    # #     spamwriter = csv.writer(csvfile, delimiter=' ',
    # #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # #     spamwriter.writerow([cur_link_precision, cur_ganunet_precision, cur_counet_precision])
    # # fig = plt.figure()
    # # plt.imshow(cur_img_label_region)
    # # plt.title("ground truth label")
    # # plt.show()
    # cur_img_label_bi = cur_img_label[:, :, 0] > 0
    # precision_counet = tp_pixel_num_sum_counet/cur_img_counet.sum()
    # recall_counet = tp_pixel_num_sum_counet/cur_img_label_bi.sum()
    # fscore_counet = 2 * precision_counet * recall_counet / (precision_counet + recall_counet)
    #
    # precision_link = tp_pixel_num_sum_link/cur_img_link.sum()
    # recall_link = tp_pixel_num_sum_link/cur_img_label_bi.sum()
    # fscore_link = 2 * precision_link * recall_link / (precision_link + recall_link)
    # #
    #
    # # precision_counet = tp_pixel_num_sum_counet/cur_img_counet.sum()
    # # recall_counet = tp_pixel_num_sum_counet/cur_img_label_bi.sum()
    # # fscore_counet = 2 * precision_counet * recall_counet / (precision_counet + recall_counet)
    # # print(tp_pixel_num_sum_counet/cur_img_counet.sum(), tp_pixel_num_sum_counet/cur_img_label_bi.sum(), fscore_counet)
    # # print(tp_pixel_num_sum_counet / cur_img_counet.sum(), tp_pixel_num_sum_counet / cur_img_label_bi.sum(), fscore_counet)
    # # print(precision_link, recall_link, fscore_link)
    # # print(precision_counet, recall_counet, fscore_counet)
    # # precision_link, recall_link, fscore_link = accurate(cur_img_link, cur_img_label_bi)
    # # fscore_link_sum = fscore_link + fscore_link_sum
    # # precision_ganunet, recall_ganunet, fscore_ganunet = accurate(cur_img_ganunet, cur_img_label_bi)
    # # fscore_ganunet_sum = fscore_ganunet + fscore_ganunet_sum
    # # precision_counet, recall_counet, fscore_counet = accurate(cur_img_counet, cur_img_label_bi)
    # # fscore_counet_sum = fscore_counet + fscore_counet_sum
    # # print(precision_link, precision_ganunet, precision_counet)
    # # print(recall_link, recall_ganunet, recall_counet)
    # # print(fscore_link, fscore_ganunet, fscore_counet)

# evaluate_one_pic(1)
pool = Pool(processes=7)
for i in range(1, 17):
    pool.apply_async(evaluate_one_pic, args=(i,))
pool.close()
pool.join()
# for i in range(1, 17):
#     evaluate_one_pic(i)
# link: 55, 36, 56, 38

# print(precision_link, recall_link, fscore_link)
# print(precision_ganunet, recall_ganunet, fscore_ganunet)
# print(precision_counet, recall_counet, fscore_counet)
# print(fscore_link_sum/4, fscore_ganunet_sum/4, fscore_counet_sum/4)

