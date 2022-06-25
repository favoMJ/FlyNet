
import os
import time
import argparse

import torch
import torch.nn.functional as F


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# args
parse = argparse.ArgumentParser()
parse.add_argument('--ckpt', dest='ckpt', type=str, default='./snapshots/AttaNet_light_706.pth',)
args = parse.parse_args()
use_boundary_2 = False
use_boundary_4 = False
use_boundary_8 = True
use_boundary_16 = False
use_conv_last = False
n_classes = 19

backbone = 'STDCNet813'
methodName = 'STDC1-Seg'
inputSize = 512
inputScale = 50
inputDimension = (1, 3, 512, 1024)

from models.model_stages import BiSeNet
from models.flynet import FlyNet
# net = BiSeNet(backbone=backbone, n_classes=n_classes,
#                 use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4,
#                 use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16,
#                 input_size=inputSize, use_conv_last=use_conv_last)#68
net = FlyNet(19,is_training=False)#89 83
# define model
# net.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
net.cuda()
net.eval()

input_t = torch.Tensor(1, 3, 1024, 2048).cuda()

print("start warm up")
for i in range(100):
    net(input_t)
print("warm up done")

start_ts = time.time()
for i in range(500):
    input = F.interpolate(input_t, (512, 1024), mode='nearest')
    net(input)
end_ts = time.time()

t_cnt = end_ts - start_ts
print("=======================================")
print("FPS: %f" % (500 / t_cnt))
print("Inference time %f ms" % (t_cnt/500*1000))