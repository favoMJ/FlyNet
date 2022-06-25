#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from models.model_stages import BiSeNet
from cityscapes import CityScapes

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
from utils import save_predict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
class MscEvalV0(object):

    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label , name) in diter:

            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]

            imgs = imgs.cuda()

            N, C, H, W = imgs.size()
            new_hw = [int(H*self.scale), int(W*self.scale)]

            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)

            logits = net(imgs)[0]

            logits = F.interpolate(logits, size=size,
                    mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes).float()
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()

def evaluatev0(respth='./pretrained', dspth='./data', backbone='CatNetSmall', scale=0.75, use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False):
    print('scale', scale)
    print('use_boundary_2', use_boundary_2)
    print('use_boundary_4', use_boundary_4)
    print('use_boundary_8', use_boundary_8)
    print('use_boundary_16', use_boundary_16)
    ## dataset
    batchsize = 1
    n_workers = 1
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    n_classes = 19
    print("backbone:", backbone)
    net = BiSeNet(backbone=backbone, n_classes=n_classes,
     use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4,
     use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16,
     use_conv_last=use_conv_last)
    net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()


    with torch.no_grad():
        single_scale = MscEvalV0(scale=scale)
        mIOU = single_scale(net, dl, 19)
    logger = logging.getLogger()
    print(mIOU)
    logger.info('mIOU is: %s\n', mIOU)


class MscEvalV1(object):

    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        ## evaluate

        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label , name) in diter:
            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            imgs = imgs.cuda()

            N, C, H, W = imgs.size()
            new_hw = [int(H * self.scale), int(W * self.scale)]

            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)

            logits = net(imgs)[0]
            logits = F.interpolate(logits, size=size,
                    mode='bilinear', align_corners=True)

            output = logits.cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            save_predict(output, None, name[0],'cityscapes','./server',
                         output_grey=True, output_color=False, gt_color=False)


def evaluatev1(respth='./pretrained', dspth='./data', backbone='CatNetSmall', scale=0.75, use_boundary_2=False,
               use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False):
    print('scale', scale)
    print('use_boundary_2', use_boundary_2)
    print('use_boundary_4', use_boundary_4)
    print('use_boundary_8', use_boundary_8)
    print('use_boundary_16', use_boundary_16)
    print(respth)
    ## dataset
    batchsize = 1
    n_workers = 1
    dsval = CityScapes(dspth, mode='test')
    dl = DataLoader(dsval,
                    batch_size=batchsize,
                    shuffle=False,
                    num_workers=n_workers,
                    drop_last=False)
    from models.flynet import FlyNet
    n_classes = 19
    print("backbone:", backbone)
    net = FlyNet(n_classes,is_training=False)
    net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()

    with torch.no_grad():
        single_scale = MscEvalV1(scale=scale)
        mIOU = single_scale(net, dl, 19)
    logger = logging.getLogger()
    logger.info('mIOU is: %s\n', mIOU)

if __name__ == "__main__":
    log_dir = 'evaluation_logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # setup_logger(log_dir)
    
    # STDC1-Seg50 mIoU 0.7222
    evaluatev1('./checkpoints/train_STDC1-Seg/pths/model_maxmIOU50.pth', dspth='/home/cv/tsx/data/cityscapes', backbone='STDCNet813', scale=0.5,
    use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
    #evaluatev0('./checkpoints/model_maxmIOU75.pth', dspth='/home/cv/tsx/data/cityscapes', backbone='STDCNet813', scale=0.5,
    #           use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
    #STDC1-Seg75 mIoU 0.7450
    # evaluatev0('./checkpoints/STDC1-Seg/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet813', scale=0.75, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)


    #STDC2-Seg50 mIoU 0.7424
    # evaluatev0('./checkpoints/STDC2-Seg/model_maxmIOU50.pth', dspth='./data', backbone='STDCNet1446', scale=0.5, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    #STDC2-Seg75 mIoU 0.7704
    # evaluatev0('./checkpoints/STDC2-Seg/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75,
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

   

