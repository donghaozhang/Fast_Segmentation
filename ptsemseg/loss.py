import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
DEBUG=False
def log(s):
    if DEBUG:
        print(s)

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0

    target = target[mask]
    # print('log_p ', log_p.size())
    # print('target ', target.size())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    # target_test = target.view(-1, 1)
    # logpt = log_p.gather(1, target_test)
    # print('loss sum logpt', logpt.sum())
    # print('loss original', loss)


    # if size_average:
    #     loss /= mask.data.sum()
    return loss

def cross_entropy3d_new(input, target, weight=None, size_average=True):

    log('The input size is {}'.format(input.size()))
    log('The output size is {}'.format(target.size()))
    # input: (n, c, h, w, z), target: (n, h, w, z)
    n, c, h, w, z = input.size()
    # log_p: (n, c, h, w, z)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w*z, c)
    log_p = log_p.permute(0, 4, 3, 2, 1).contiguous().view(-1, c) # make class dimension last dimension
    log_p = log_p[target.view(n * h * w *z, 1).repeat(1, c) >= 0] # this looks wrong -> Should rather be a one-hot vector
    log_p = log_p.view(-1, c)
    # target: (n*h*w*z,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    return loss

def cross_entropy3d(input, target, weight=None, size_average=True):

    log('The size of input is {}'.format(input.size()))
    input = F.log_softmax(input, dim=1)
    log('LOSS=>CrossEntropy3D=>input.size():{} target.size():{}'.format(input.size(), target.size()))
    loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average)
    return loss(input, target)