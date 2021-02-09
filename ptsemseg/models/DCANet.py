import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from ptsemseg.models.utils import *
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DCABlock(nn.Module):
    def __init__(self, channel):
        super(DCABlock,self).__init__()
        #global pooling
        self.conv_gp = nn.Conv2d(channel, channel, 1, 1, 0,bias=False)
        self.bn_gp = nn.BatchNorm2d(channel)

        self.conv = nn.Conv2d(channel, channel, 1 , 1, bias=False)
        self.bn = nn.BatchNorm2d(channel)

        #self.dilate0 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0, bias=False)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1, bias=False)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, bias=False)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4, bias=False)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8, bias=False)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16, bias=False)

        self.conv1 = nn.Conv2d(1024, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)


        self._init_weight()

    def forward(self, input):
        x_master = self.conv(input)
        x_master = self.bn(x_master)

        x_gp = nn.AvgPool2d(input.shape[2:])(input)
        x_gp = self.conv_gp(x_gp)
        x_gp = self.bn_gp(x_gp)

        dilate1_out = self.bn(self.dilate1(input))
        dilate1_out_relu = nonlinearity(dilate1_out)
        dilate2_out = self.bn(self.dilate2((dilate1_out_relu)))
        dilate2_out_relu = nonlinearity(dilate2_out)
        dilate3_out = self.bn(self.dilate3((dilate2_out_relu)))
        dilate3_out_relu = nonlinearity(dilate3_out)
        dilate4_out = self.bn(self.dilate4((dilate3_out_relu)))
        dilate4_out_relu = nonlinearity(dilate4_out)
        dilate5_out = self.bn(self.dilate5((dilate4_out_relu)))


        out1 = nonlinearity(dilate2_out - dilate1_out)
        out2 = nonlinearity(dilate3_out - dilate2_out)
        out3 = nonlinearity(dilate4_out - dilate3_out)
        out4 = nonlinearity(dilate5_out - dilate4_out)

        out = torch.cat((out1, out2,out3, out4), dim=1)
        x = self.conv1(out)
        x = self.bn1(x)
        x = nonlinearity(x)
        # x = out1 + out2 + out3 + out4

        finalout = nonlinearity(x_master*x + x_gp)

        return finalout

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, n_classes):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(n_filters, n_filters, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.last_conv = nn.Sequential(nn.Conv2d(in_channels + n_filters, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, n_classes-1, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = nonlinearity(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DCANet(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(DCANet, self).__init__()

        filters = [64, 128, 256, 512] #ResNet34
        #filters = [64, 256, 512, 1024] #ResNet50
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3


        # Auxiliary layers for training
        self.convbnrelu2_aux = conv2DBatchNormRelu(
            in_channels=128, k_size=3, n_filters=64, padding=1, stride=1, bias=False
        )
        self.aux_cls = nn.Conv2d(64, n_classes-1, 1, 1, 0)


        self.dblock = DCABlock(256) #ResNet34
        #self.dblock = DCABlock(1024) #ResNet50


        self.decoder = DecoderBlock(filters[2], filters[0], n_classes)



    def forward(self, input):
        # Encoder
        x = self.firstconv(input)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)

        # Auxiliary layers for training
        x_aux = self.convbnrelu2_aux(e2)
        # x_aux = self.dropout(x_aux)
        x_aux = self.aux_cls(x_aux)
        x_aux = F.sigmoid(x_aux)


        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)

        # Decoder
        decoder = self.decoder(e3, e1)
        out = F.interpolate(decoder, size=input.size()[2:], mode='bilinear', align_corners=True)


        out = F.sigmoid(out)
        # return F.sigmoid(out)
        if self.training:
            return x_aux, out
        else:  # eval mode
            return out


