import torch.nn as nn
import torch
import numpy as np
import time
from thop import profile
from thop import clever_format
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ptsemseg.models.fusionmodel import *
from ptsemseg.models.AMTUnet import *
from ptsemseg.models.ESFNet import *
from ptsemseg.models.RCNNUNet import *
from ptsemseg.models.CasNet import *
from ptsemseg.models.BMANet import *
from ptsemseg.models.difpyramidnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.SAR_unet import *
from ptsemseg.models.utils import *
from torchvision.models import *
from ptsemseg.models.ResUnet import *
from ptsemseg.models.fcn import *
from ptsemseg.models.segnet import *
from ptsemseg.models.AMTUnet import *

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
model =DifPyramidNet()
# model = torch.nn.DataParallel(model_b)
input = torch.randn(1,3,224,224)
flop, para = profile(model, inputs = (input, ))
flop, para = clever_format([flop, para], "%.3f")
# print("%.2fM"%(flop/1e6), "%.2fM"%(para/1e6))
print(flop, para)

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

model.eval()
model = model.cuda()

input = torch.randn((1,3,512,512), device=device)
for _ in range(10):
    model(input)

torch.cuda.synchronize()
torch.cuda.synchronize()
t_start = time.time()
for _ in range(500):
    model(input)
torch.cuda.synchronize()
torch.cuda.synchronize()
elapsed_time = time.time() - t_start