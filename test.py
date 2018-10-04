import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as  plt

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import scores

def test(args):



    # Setup image
    # print("Read Input Image from : ", args.img_path)
	#
    # data_loader = get_loader(args.dataset)
    # data_path = get_data_path(args.dataset)
    # loader = data_loader(data_path, is_transform=True)
    # n_classes = loader.n_classes
	#
    # img = img[:, :, ::-1]
    # img = img.astype(np.float64)
    # img -= loader.mean
    # # print(loader.img_size[0], loader.img_size[1])
    # # img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
    # img = img.astype(float) / 255.0
    # # NHWC -> NCWH
    # img = img.transpose(2, 0, 1)
    # img = np.expand_dims(img, 0)
    # img = torch.from_numpy(img).float()
	#
    # # Setup Model
    # model = torch.load(args.model_path)
    # model.eval()
	#
    # if torch.cuda.is_available():
    #     model.cuda(0)
    #     images = Variable(img.cuda(0))
    # else:
    #     images = Variable(img)
	#
    # outputs = model(images)
    # log('the size of outputs is '.format(outputs.size()))
    # log('the size of outputs is ', outputs.data.max(1)[1].size())
    # pred = np.squeeze(outputs.data.cpu().numpy(), axis=0)
    # pred = pred[0,:,:] > pred[1,:,:]
    # pred = pred * 255
    # pred = outputs.data.cpu().numpy()
    log(''pred.shape)
    #print('the size of pred is ', pred.shape())
    # fig = plt.figure()
    # plt.imshow(pred[1,:,:])
    # plt.title('testing result')
    # plt.show()
    pred = pred * 255

    misc.imsave(args.out_path, pred[0,:,:])
    print("Segmentation Mask Saved at: ", args.out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None, 
                        help='Path of the output segmap')
    args = parser.parse_args()
    test(args)
