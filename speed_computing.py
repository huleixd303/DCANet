import torch.nn as nn
import torch
import numpy as np
import time
from thop import profile
from thop import clever_format
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ptsemseg.models.fusionmodel import *
from ptsemseg.models.ESFNet import *
from ptsemseg.models.RCNNUNet import *
from ptsemseg.models.CasNet import *
from ptsemseg.models.BMANet import *
from ptsemseg.models.difpyramidnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.SAR_unet import *
from ptsemseg.models.utils import *
from ptsemseg.models.segnet import *
from torchvision.models import *
from ptsemseg.models.fcn import *
from ptsemseg.models.AMTUnet import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

model = DifPyramidNet()
model.eval()
model = model.cuda()
input_size = (1,3,512,512)
input = torch.randn(input_size, device=device)

for _ in range(10):
    model(input)

# logger.info('=========Speed Testing=========')
torch.cuda.synchronize()
torch.cuda.synchronize()
t_start = time.time()
for _ in range(1000):
    model(input)
torch.cuda.synchronize()
torch.cuda.synchronize()
elapsed_time = time.time() - t_start
print(elapsed_time)
print("FPS: %f"%(1.0/(elapsed_time/1000)))
# logger.info(
#     'Elapsed time: [%.2f s / %d iter]' % (elapsed_time, iteration))
# logger.info('Speed Time: %.2f ms / iter    FPS: %.2f' % (
#     elapsed_time / iteration * 1000, iteration / elapsed_time))
