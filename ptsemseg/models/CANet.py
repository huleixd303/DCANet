import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from ptsemseg.models.utils import *
from functools import partial
from ptsemseg.models.fusionmodel import *
from torch.nn import init
from ptsemseg.models.Carafe import *
import copy
nonlinearity = partial(F.relu, inplace=True)

class Decoder_new(nn.Module):
    def __init__(self, channel, out):
        super(Decoder_new, self).__init__()

        self.upsample = CARAFE(channel, c_mid=channel//2, scale=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(channel, out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        x = self.upsample(input)
        x = self.conv_bn_relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1,bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

        # self.conv4 = nn.Conv2d(n_filters*2, n_filters, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        # x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x

class DifPyramidBlock(nn.Module):
    def __init__(self, channel):
        super(DifPyramidBlock,self).__init__()


        #global pooling
        # self.conv_gp = nn.Conv2d(channel, channel, 1, 1, 0,bias=False)
        # self.bn_gp = nn.BatchNorm2d(channel)
        # self.channels_cond = channel

        self.conv = nn.Conv2d(channel, channel, 1 , 1, bias=False)
        self.bn = nn.BatchNorm2d(channel)

        self.bn_d1 = nn.BatchNorm2d(channel)
        self.bn_d2 = nn.BatchNorm2d(channel)
        self.bn_d3 = nn.BatchNorm2d(channel)
        self.bn_d4 = nn.BatchNorm2d(channel)
        self.bn_d5 = nn.BatchNorm2d(channel)

        #self.dilate0 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0, bias=False)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1, bias=False)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, bias=False)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4, bias=False)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8, bias=False)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16, bias=False)

        # self.weights = nn.Parameter(torch.randn(2))
        # self.nums = 2
        # self._reset_parameters()

        self.conv1 = nn.Conv2d(channel*3, channel, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)



        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def forward(self, input):

        dilate1_out = self.bn(self.dilate1(input))
        # dilate1_out_relu = nonlinearity(dilate1_out)
        dilate2_out = self.bn(self.dilate2((dilate1_out)))
        # dilate2_out_relu = nonlinearity(dilate2_out)
        dilate3_out = self.bn(self.dilate3((dilate2_out)))
        # # dilate3_out_relu = nonlinearity(dilate3_out)
        dilate4_out = self.bn(self.dilate4((dilate3_out)))
        # # dilate4_out_relu = nonlinearity(dilate4_out)
        # dilate5_out = self.bn(self.dilate5((dilate4_out)))
        # dilate5_out_relu = nonlinearity(dilate5_out)

        out1 = nonlinearity((dilate2_out - dilate1_out))
        out2 = nonlinearity((dilate3_out - dilate2_out))
        out3 = nonlinearity((dilate4_out - dilate3_out))
        # out4 = nonlinearity((dilate5_out - dilate4_out))

        #intergrating discriminative feature
        # out = out1 + out2 + out3 + out4
        out = torch.cat((out1, out2, out3), dim=1)

        x = nonlinearity(self.bn1(self.conv1(out)))
        #
        #
        # # finalout = self.weights[0] * x + self.weights[1] * input
        finalout = x

        return finalout

class CANet(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(CANet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2

        self.block = nn.Sequential(
            nn.Conv2d(filters[0]//2, filters[0]//4, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]//4),
            nn.ReLU(True),

            nn.Conv2d(filters[0]//4, n_classes-1, 1)
        )

        self.decoder = Decoder_new(channel=filters[1], out=filters[0]//2)


    def forward(self, x):
        # Encoder
        size = x.size()[2:]
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        out = self.decoder(e2)
        out = self.block(out)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)

        return F.sigmoid(out)
