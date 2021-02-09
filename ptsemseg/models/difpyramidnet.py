"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from ptsemseg.models.utils import *
from functools import partial
from ptsemseg.models.fusionmodel import *
from torch.nn import init
import copy

nonlinearity = partial(F.relu, inplace=True)

class SELayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.d = max(int(out_planes//reduction), 32)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, self.d),
            nn.ReLU(inplace=True),
            nn.Linear(self.d, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(fm)
        fm = x1 * channel_attetion + x2

        return fm

class ChannelAttention_1(nn.Module):
    def __init__(self, in_planes, out_planes, reduction = 8):
        super(ChannelAttention_1, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        # fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(x2)
        fm = x1 * channel_attetion + x2

        return fm

class CatAttention(nn.Module):
    def __init__(self, in_planes, out_planes, reduction):
        super(CatAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU,
            nn.Conv2d(out_planes, 1, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(fm)

        fm = x1 * channel_attetion

        return fm

class PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels//8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels//8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out

class SpatialAttention(nn.Module):
    def __init__(self,F_l,F_h,F_int):
        super(SpatialAttention,self).__init__()


        self.W_h = nn.Sequential(
            nn.Conv2d(F_h, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_h(g)
        x1 = self.W_l(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class SpatialAttention_1(nn.Module):
    def __init__(self,channel_low, channel_high):
        super(SpatialAttention_1, self).__init__()

        self.h_m = nn.Sequential(
            nn.Conv2d(channel_high, channel_high, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel_high)
        )

        self.l_m = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel_low)
        )
        # self.psi = nn.Sequential(
        #     nn.Conv2d(channel_mid, 1, 1, 1, 0, bias=False),
        #     # nn.BatchNorm2d(1),
        #     nn.Sigmoid()
        # )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, lowfm, highfm):
        h_m = self.h_m(highfm)
        l_m = self.l_m(lowfm)

        fms = nonlinearity(h_m - l_m)
        # psi = self.psi(fms) * lowfm
        # psi = self.psi(fms)

        # out = torch.cat((psi, highfm), 1)
        # finalout = self.convblock(out)

        return fms

class Conv2(nn.Module):
    def __init__(self, channel):
        super(Conv2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(channel, channel, 3, 1, 1),
        #     nn.BatchNorm2d(channel),
        #     nn.ReLU(),
        # )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        # outputs = self.conv2(outputs)
        return outputs

class LargeKernelConv(nn.Module):
    def __init__(self, in_dim, kernel_size):
        super(LargeKernelConv, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        self.conv_l1 = nn.Conv2d(in_dim, in_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(in_dim , in_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, in_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(in_dim , in_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        # self.conv = Conv2(in_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        out = x_l + x_r
        return out

class GlobalConvModule(nn.Module):
    def __init__(self, in_dim, kernel_size):
        super(GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        # kernel size had better be odd number so as to avoid alignment error
        super(GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, in_dim//2, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(in_dim//2, 1, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, in_dim//2, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(in_dim//2, 1, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        out = F.sigmoid(x_l + x_r)
        finalout = x*out

        return finalout

class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class DifPyramidBlock(nn.Module):
    def __init__(self, channel):
        super(DifPyramidBlock,self).__init__()


        #global pooling
        self.conv_gp = nn.Conv2d(channel, channel, 1, 1, 0,bias=False)
        self.bn_gp = nn.BatchNorm2d(channel)
        self.channels_cond = channel

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
    #
    # def _reset_parameters(self):
    #     init.constant_(self.weights, 1 / self.nums)

    def forward(self, input):
        # x_master = self.conv(input)
        # x_master = self.bn(x_master)

        # size = input.size()[2:]
        # x_gp = nn.AvgPool2d(input.shape[2:])(input)
        # x_gpb = nn.AvgPool2d(input.shape[2:])(input).view(input.shape[0], self.channels_cond, 1, 1)
        # x_gpb = self.conv_gp(x_gpb)
        # x_gpb = nonlinearity(self.bn_gp(x_gpb))
        # x_gp = nonlinearity(x_gp)
        # x_gp = F.interpolate(x_gp, size, mode='bilinear', align_corners=True)


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


class Dblock_mod(nn.Module):
    def __init__(self, channel):
        super(Dblock_mod, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.scaleattetion = ScaleAttentionModule()
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        self.conv_local = conv2DBatchNormRelu(
            in_channels=256*4, k_size=1, n_filters=256, padding=0, stride=1, bias=False
        )
        self.conv_glo = conv2DBatchNormRelu(
            in_channels=256, k_size=1, n_filters=256, padding=0, stride=1, bias=False
        )

        self.conv_aggreagtion = conv2DBatchNormRelu(
            in_channels=256*2, k_size=1, n_filters=256, padding=0, stride=1, bias=False
        )
        self.bn = nn.BatchNorm2d(channel)
        # self.conv = nn.Conv2d(channels, channels, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.bn(self.dilate1(x)))
        x_new = x.unsqueeze(1)
        dilate1_out_new = dilate1_out.unsqueeze(1)
        # dilate1_out_cat = torch.cat((x_new, dilate1_out_new), dim=1)
        # dilate1_out_final = self.scaleattetion(dilate1_out_cat, 256)


        dilate2_out = nonlinearity(self.bn(self.dilate2(dilate1_out)))
        # dilate1_out_final_new = dilate1_out_final.unsqueeze(1)
        dilate2_out_new = dilate2_out.unsqueeze(1)
        # dilate2_out_cat = torch.cat((dilate1_out_final_new, dilate2_out_new), dim=1)
        # dilate2_out_final = self.scaleattetion(dilate2_out_cat, 256)

        dilate3_out = nonlinearity(self.bn(self.dilate3(dilate2_out)))
        # dilate2_out_final_new = dilate2_out_final.unsqueeze(1)
        dilate3_out_new = dilate3_out.unsqueeze(1)
        # dilate3_out_cat = torch.cat((dilate2_out_final_new, dilate3_out_new), dim=1)
        # dilate3_out_final = self.scaleattetion(dilate3_out_cat, 256)

        dilate4_out = nonlinearity(self.bn(self.dilate4(dilate3_out)))
        # dilate3_out_final_new = dilate3_out_final.unsqueeze(1)
        dilate4_out_new = dilate4_out.unsqueeze(1)
        # dilate4_out_cat = torch.cat((dilate3_out_final_new, dilate4_out_new), dim=1)
        # dilate4_out_final = self.scaleattetion(dilate4_out_cat, 256)

        # local_out = torch.cat((dilate1_out_final, dilate2_out_final, dilate3_out_final, dilate4_out_final), dim =1)
        # local_out = self.conv_local(local_out)

        glo_out_cat = torch.cat((x_new, dilate1_out_new, dilate2_out_new, dilate3_out_new, dilate4_out_new), dim=1)
        glo_out = self.scaleattetion(glo_out_cat, 256)
        glo_out = self.conv_glo(glo_out)


        # finalout = torch.cat((local_out, glo_out), dim=1)
        # finalout = self.conv_aggreagtion(finalout)
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        # out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return glo_out

class ScaleAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self):
        super(ScaleAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, dim):
        batch_size, scale, channel, height, width = x.size()
        num_fuse = channel//dim
        feat_a = x.view(batch_size, -1, channel * height * width)
        feat_a_transpose = x.view(batch_size, -1, channel * height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        # attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        # attention = self.softmax(attention_new)
        attention = self.softmax(attention)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, channel, height, width)
        out = self.beta * feat_e + x
        # out = feat_e + x
        finalout = 0
        for i in range(num_fuse):
            finalout = finalout + out[:, i, :, :, :]

        # finalout = out[:, 0, :, :, :] + out[:, 1, :, :, :]

        return finalout


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

        # sup_out = self.relu3(x - lowfm)
        # out = torch.cat((sup_out, lowfm), dim =1)
        # out = self.conv4(out)
        # out= nonlinearity(self.norm3(out))



        return x

class spatialgate(nn.Module):
    def __init__(self, channel):
        super(spatialgate,self).__init__()

        self.conv_reduction = nn.Conv2d(channel, 32, 1, bias=False)
        self.bn_reducton = nn.BatchNorm2d(32)

        self.dilate1 = nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2, bias=False)
        self.dilate2 = nn.Conv2d(32, 32, kernel_size=3, dilation=5, padding=5, bias=False)

        self.psi = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, highfm, lowfm):
        low_out = self.conv_reduction(lowfm)
        low_out = self.bn_reducton(low_out)
        low_out = nonlinearity(low_out)
        low_out = nonlinearity(self.bn_reducton(self.dilate1(low_out)))
        low_out = nonlinearity(self.bn_reducton(self.dilate2(low_out)))
        low_psi = self.psi(low_out)

        high_out = self.conv_reduction(highfm)
        high_out = self.bn_reducton(high_out)
        high_out = nonlinearity(high_out)
        high_psi = self.psi(high_out)

        out = (1 + low_psi) * lowfm + (1 - low_psi) * high_psi * highfm

        return  out


class upfusion(nn.Module):
    def __init__(self, channel):
        super(upfusion, self).__init__()

        self.conv_highfm = nn.Conv2d(channel, channel, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.conv_gp = nn.Conv2d(channel, channel, 1, bias=False)
        self.conv_cat = nn.Conv2d(channel*2, channel, 1, bias=False)

    def forward(self, highfm, lowfm):
        x_gp = nn.AvgPool2d(highfm.shape[2:])(highfm)
        x_gp = self.conv_gp(x_gp)
        x_gp = self.bn(x_gp)

        x = self.bn(self.conv_highfm(highfm))
        sub_out = nonlinearity(x - lowfm)
        out = torch.cat((sub_out, lowfm), dim = 1)
        out = self.conv_cat(out)
        out = nonlinearity(self.bn(out))

        out = nonlinearity(out + x_gp)

        return out


class DifPyramidNet_3(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(DifPyramidNet_3, self).__init__()

        filters = [64, 128, 256, 512] #ResNet34
        # filters = [64, 256, 512, 1024] #ResNet50
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3


        # Auxiliary layers for training
        # self.convbnrelu2_aux = conv2DBatchNormRelu(
        #     in_channels=128, k_size=3, n_filters=64, padding=1, stride=1, bias=False
        # )
        # self.aux_cls = nn.Conv2d(64, n_classes - 1, 1, 1, 0)


        self.dblock = DifPyramidBlock(256) #ResNet34
        # self.dblock = DifPyramidBlock(512) #ResNet50
        self.outconv_e3_t2d = conv2DBatchNormRelu(
            in_channels=256, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )
        self.outconv_e3_b2u = conv2DBatchNormRelu(
            in_channels=256, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.outconv_d3_t2d = conv2DBatchNormRelu(
            in_channels=128, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )
        self.outconv_d3_b2u = conv2DBatchNormRelu(
            in_channels=128, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )
        self.convcat3_t2d = nn.Conv2d(2, 1, 1, bias=False)
        self.convcat3_b2u = nn.Conv2d(2, 1, 1, bias=False)
        # self.upfusion3 = spatialgate(filters[1])
        # self.channelattention_decoder3 = SpatialAttention(filters[1], filters[1], filters[1]//2)
        # self.spatialattention_e2 = GlobalConvModule(filters[1], (15,15))
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.outconv_d2_t2d = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )
        self.outconv_d2_b2u = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )
        self.convcat2_t2d = nn.Conv2d(3, 1, 1, bias=False)
        self.convcat2_b2u = nn.Conv2d(3, 1, 1, bias=False)
        # self.upfusion2 = spatialgate(filters[0])
        # self.channelattention_decoder2 = SpatialAttention(filters[0], filters[0], filters[0]//2)
        # self.spatialattention_e1 = GlobalConvModule(filters[0], (15, 15))
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.outconv_d1_t2d = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )
        self.outconv_d1_b2u = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )
        self.convcat1_t2d = nn.Conv2d(4, 1, 1, bias=False)
        self.convcat1_b2u = nn.Conv2d(4, 1, 1, bias=False)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1,bias=False)

        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.finalrelu2 = nonlinearity
        # self.dropout = nn.Dropout(0.1)
        self.finalconv3_t2d = nn.Conv2d(32, n_classes-1, 3, padding=1)
        self.finalconv3_b2u = nn.Conv2d(32, n_classes - 1, 3, padding=1)
        self.convcat0_t2d = nn.Conv2d(5, 1, 1, bias=False)
        self.convcat0_b2u = nn.Conv2d(5, 1, 1, bias=False)

        # self.refunet = RefUnet(1, 64)
        self.fuse = FusionLayer()
        self.conv1 = Conv2(1)
        self.layencoding = LayerEncoding()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()



    def forward(self, x):
        # prob, back = list(), list()
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)
        e3_aux_t2d = self.outconv_e3_t2d(e3)
        e3_aux_t2d = F.interpolate(e3_aux_t2d, scale_factor=16.0, mode='bilinear', align_corners=True)
        e3_aux_b2u = self.outconv_e3_b2u(e3)
        e3_aux_b2u = F.interpolate(e3_aux_b2u, scale_factor=16.0, mode='bilinear', align_corners=True)
        e3_aux_t2d_sig = F.sigmoid(e3_aux_t2d)
        # # back.append(e3_aux_t2d)
        #
        # e3_aux_t2d_sig = F.sigmoid(e3_aux_t2d)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d3_aux_t2d = self.outconv_d3_t2d(d3)
        d3_aux_t2d = F.interpolate(d3_aux_t2d, scale_factor=8.0, mode='bilinear', align_corners=True)
        d3_aux_b2u = self.outconv_d3_b2u(d3)
        d3_aux_b2u = F.interpolate(d3_aux_b2u, scale_factor=8.0, mode='bilinear', align_corners=True)

        # back.append(d3_aux_t2d)
        # fusion_d3 = [e3_aux_t2d]
        # d3_aux_t2d_encode = self.layencoding(d3_aux_t2d, fusion_d3)
        d3_aux_t2d = torch.cat((d3_aux_t2d, e3_aux_t2d), dim=1)
        d3_aux_t2d = self.convcat3_t2d(d3_aux_t2d)
        d3_aux_t2d_sig = F.sigmoid(d3_aux_t2d)
        # fusion_d3 = [d3_aux_t2d, e3_aux_t2d]
        # fusion_d3_out = self.fuse(fusion_d3)
        # fusion_d3_out_sig = F.sigmoid(fusion_d3_out)
        # d3 = self.upfusion3(d3, e2)
        # d3 = self.channelattention_decoder3(e2, d3)
        # e2_sp = self.spatialattention_e2(e2)
        d2 = self.decoder2(d3) + e1
        d2_aux_t2d = self.outconv_d2_t2d(d2)
        d2_aux_t2d = F.interpolate(d2_aux_t2d, scale_factor=4.0, mode='bilinear', align_corners=True)
        d2_aux_b2u = self.outconv_d2_b2u(d2)
        d2_aux_b2u = F.interpolate(d2_aux_b2u, scale_factor=4.0, mode='bilinear', align_corners=True)


        # back.append(d2_aux_t2d)
        # fusion_d2 = [d3_aux_t2d, e3_aux_t2d]
        # d2_aux_t2d_encode = self.layencoding(d2_aux_t2d, fusion_d2)
        d2_aux_t2d = torch.cat((d3_aux_t2d, d2_aux_t2d, e3_aux_t2d), dim=1)
        d2_aux_t2d = self.convcat2_t2d(d2_aux_t2d)
        d2_aux_t2d_sig = F.sigmoid(d2_aux_t2d)
        # fusion_d2 = [d3_aux_t2d, d2_aux_t2d]
        # fusion_d2_out = self.fuse(fusion_d2)
        # fusion_d2_out_sig = F.sigmoid(fusion_d2_out)
        # d2 = self.upfusion2(d2, e1)
        # d2 = self.channelattention_decoder2(e1, d2)
        d1 = self.decoder1(d2)
        d1_aux_t2d = self.outconv_d1_t2d(d1)
        d1_aux_t2d = F.interpolate(d1_aux_t2d, scale_factor=2.0, mode='bilinear', align_corners=True)
        d1_aux_b2u = self.outconv_d1_b2u(d1)
        d1_aux_b2u = F.interpolate(d1_aux_b2u, scale_factor=2.0, mode='bilinear', align_corners=True)
        # back.append(d1_aux_t2d)
        # fusion_d1 = [d3_aux_t2d, d2_aux_t2d, e3_aux_t2d]
        # d1_aux_t2d_encode = self.layencoding(d1_aux_t2d, fusion_d1)
        d1_aux_t2d = torch.cat((d3_aux_t2d, d2_aux_t2d, d1_aux_t2d, e3_aux_t2d), dim=1)
        d1_aux_t2d = self.convcat1_t2d(d1_aux_t2d)
        d1_aux_t2d_sig = F.sigmoid(d1_aux_t2d)
        # fusion_d1 = [d1_aux_t2d, d2_aux_t2d]
        # fusion_d1_out = self.fuse(fusion_d1)
        # fusion_d1_out_sig = F.sigmoid(fusion_d1_out)

        # Final Classification
        out = self.finaldeconv1(d1)
        #out = self.bn(out)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        #out = self.bn(out)
        out = self.finalrelu2(out)

        #out = self.dropout(out)
        # out = self.finalconv3(out)
        out_b2u = self.finalconv3_b2u(out)
        out_t2d = self.finalconv3_t2d(out)
        # back.append(out_t2d)
        # fusion_d0 = [d3_aux_t2d, d2_aux_t2d, d1_aux_t2d, e3_aux_t2d]
        # out_t2d_encode = self.layencoding(out, fusion_d0)
        out_b2u_sig = F.sigmoid(out_b2u)
        out_t2d = torch.cat((d3_aux_t2d, d2_aux_t2d, d1_aux_t2d, out_t2d, e3_aux_t2d), dim=1)
        out_t2d = self.convcat0_t2d(out_t2d)
        out_t2d_sig = F.sigmoid(out_t2d)
        # fusion_d0 = [d1_aux_t2d, out_t2d]
        # fusion_d0_out = self.fuse(fusion_d0)
        # fusion_d0_out_sig = F.sigmoid(fusion_d0_out)
        # out = self.refunet(out)
        # out = F.sigmoid(out)

        d1_aux_b2u = torch.cat((out_b2u, d1_aux_b2u), dim=1)
        d1_aux_b2u = self.convcat3_b2u(d1_aux_b2u)
        d1_aux_b2u_sig = F.sigmoid(d1_aux_b2u)

        d2_aux_b2u = torch.cat((out_b2u, d1_aux_b2u, d2_aux_b2u), dim=1)
        d2_aux_b2u = self.convcat2_b2u(d2_aux_b2u)
        d2_aux_b2u_sig = F.sigmoid(d2_aux_b2u)

        d3_aux_b2u = torch.cat((out_b2u, d1_aux_b2u, d2_aux_b2u, d3_aux_b2u), dim=1)
        d3_aux_b2u = self.convcat1_b2u(d3_aux_b2u)
        d3_aux_b2u_sig = F.sigmoid(d3_aux_b2u)

        e3_aux_b2u = torch.cat((out_b2u, d1_aux_b2u, d2_aux_b2u, d3_aux_b2u, e3_aux_b2u), dim=1)
        e3_aux_b2u = self.convcat0_b2u(e3_aux_b2u)
        e3_aux_b2u_sig = F.sigmoid(e3_aux_b2u)

        # outfusion = [d3_aux_t2d, d2_aux_t2d, d1_aux_t2d,
        #              out_t2d]
        # outfusion = [out_b2u, d1_aux_b2u, d2_aux_b2u, d3_aux_b2u,
        #              out_t2d, d1_aux_t2d]
        # outfusion = [out_t2d, out_b2u]
        outfusion = [out_b2u, d1_aux_b2u, d2_aux_b2u, d3_aux_b2u, e3_aux_b2u,
                     out_t2d, d1_aux_t2d]
        # outfusion = [d2_aux_b2u, d3_aux_b2u, e3_aux_b2u,
        #              out_t2d, d1_aux_t2d]
        finalrefuse = self.fuse(outfusion)
        finalrefuse = F.sigmoid(finalrefuse)
        if self.training:
            # return d3_aux_b2u_sig, d2_aux_b2u_sig, d1_aux_b2u_sig, out_b2u_sig, d3_aux_t2d_sig, \
            #        d2_aux_t2d_sig, d1_aux_t2d_sig, out_t2d_sig, finalrefuse
            return e3_aux_b2u_sig, d3_aux_b2u_sig, d2_aux_b2u_sig, d1_aux_b2u_sig, out_b2u_sig, e3_aux_t2d_sig, d3_aux_t2d_sig, \
                   d2_aux_t2d_sig, d1_aux_t2d_sig, out_t2d_sig, finalrefuse
            # return e3_aux_b2u_sig, d3_aux_b2u_sig, d2_aux_b2u_sig, e3_aux_t2d_sig, d3_aux_t2d_sig, \
            #        d2_aux_t2d_sig, d1_aux_t2d_sig, out_t2d_sig, finalrefuse
            # return e3_aux_t2d_sig, d3_aux_t2d_sig, d2_aux_t2d_sig, d1_aux_t2d_sig, out_t2d_sig, \
            #        fusion_d1_out_sig, fusion_d0_out_sig
            # return e3_aux_t2d_sig, d3_aux_t2d_sig, d2_aux_t2d_sig, d1_aux_t2d_sig, out_t2d_sig, finalrefuse
        else:  # eval mode
            return finalrefuse

class DifPyramidNet_2(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(DifPyramidNet_2, self).__init__()

        filters = [64, 128, 256, 512] #ResNet34
        # filters = [64, 256, 512, 1024] #ResNet50
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.lay1_down_2 = DownFusionBlock(filters[0], filters[1], 2)
        self.lay1_down_4 = DownFusionBlock(filters[0], filters[2], 4)
        # self.lay1_up_2 = UpFusionBlock(filters[0], filters[0], 2)

        self.encoder2 = resnet.layer2
        self.lay2_down_2 = DownFusionBlock(filters[1], filters[2], 2)
        self.lay2_up_2 = UpFusionBlock(filters[1], filters[0], 2)
        # self.lay2_up_4 = UpFusionBlock(filters[1], filters[0], 4)

        self.encoder3 = resnet.layer3
        self.lay3_up_2 = UpFusionBlock(filters[2], filters[1], 2)
        self.lay3_up_4 = UpFusionBlock(filters[2], filters[0], 4)
        # self.dblock_up_4 = UpFusionBlock(filters[2], filters[0], 4)

        # self.lay3_up_8 = UpFusionBlock(filters[2], filters[0], 8)



        self.dblock = DifPyramidBlock(256) #ResNet34
        # self.
        # self.attention_e3 = SpatialAttention(filters[2], filters[2], filters[2] // 2)
        # self.e3_reduction = nn.Sequential(
        #     nn.Conv2d(filters[2] * 4, filters[2], 1, 1, bias=False),
        #     nn.BatchNorm2d(filters[2]),
        #     nn.ReLU(inplace=True)
        # )

        self.decoder3 = DecoderBlock(filters[2], filters[1])

        # self.attention_3_minus_e3 = SpatialAttention(filters[1], filters[1])
        # self.attention_3_minus_e2 = SpatialAttention(filters[1], filters[1])
        # self.attention_3_minus_e1 = SpatialAttention(filters[1], filters[1])
        self.channelfusion_e2_d3 = ChannelAttention(filters[1] * 2, filters[1])
        self.channelfusion_e3_d3 = ChannelAttention(filters[1] * 2, filters[1])
        self.channelfusion_e1_d3 = ChannelAttention(filters[1] * 2, filters[1])
        self.d3_reduction = nn.Sequential(
            nn.Conv2d(filters[1]*3, filters[1], 1, 1, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )


        self.decoder2 = DecoderBlock(filters[1], filters[0])

        # self.attention_2_minus_e3 = SpatialAttention(filters[0], filters[0])
        # self.attention_2_minus_e2 = SpatialAttention(filters[0], filters[0])
        # self.attention_2_minus_e1 = SpatialAttention(filters[0], filters[0])
        self.channelfusion_e2_d2 = ChannelAttention(filters[0] * 2, filters[0])
        self.channelfusion_e3_d2 = ChannelAttention(filters[0] * 2, filters[0])
        self.channelfusion_e1_d2 = ChannelAttention(filters[0] * 2, filters[0])
        # self.attention_2_minus_dblock = SpatialAttention(filters[0], filters[0])
        self.d2_reduction = nn.Sequential(
            nn.Conv2d(filters[0] * 3, filters[0], 1, 1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = DecoderBlock(filters[0], filters[0])
        # self.attention_1 = SpatialAttention(filters[0], filters[0], filters[0] // 2)
        # self.d1_reduction = nn.Sequential(
        #     nn.Conv2d(filters[0] * 3, filters[0], 1, 1, bias=False),
        #     nn.BatchNorm2d(filters[0]),
        #     nn.ReLU(inplace=True)
        # )


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1,bias=False)

        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.finalrelu2 = nonlinearity
        # self.dropout = nn.Dropout(0.1)
        self.finalconv3 = nn.Conv2d(32, n_classes-1, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3_block = self.dblock(e3)

        # Decoder

        d3 = self.decoder3(e3_block)
        # d3_fusion = e2 + self.lay1_down_2(e1) + self.lay3_up_2(e3)
        # d3_fusion_factor = self.attention_3(d3_fusion, d3)
        # d3_fusion_factor_e2 = self.attention_3_minus_e2(e2, d3)
        # d3_fusion_factor_e1 = self.attention_3_minus_e1(self.lay1_down_2(e1), d3)
        # d3_fusion_factor_e3 = self.attention_3_minus_e3(self.lay3_up_2(e3), d3)
        d3_fusion_factor_e2 = self.channelfusion_e2_d3(e2, d3)
        d3_fusion_factor_e1 = self.channelfusion_e1_d3(self.lay1_down_2(e1), d3)
        d3_fusion_factor_e3 = self.channelfusion_e3_d3(self.lay3_up_2(e3), d3)
        d3 = torch.cat(((d3_fusion_factor_e2),
                            (d3_fusion_factor_e1),
                            (d3_fusion_factor_e3)), dim=1)
        # # d3 = torch.cat((d3, e2 * (d3_fusion_factor_e2),
        # #                 self.lay1_down_2(e1) * (d3_fusion_factor_e1),
        # #                 self.lay3_up_2(e3) * (d3_fusion_factor_e3)), dim=1)
        d3 = self.d3_reduction(d3)
        # d3 = d3_fusion_factor_e3 + d3_fusion_factor_e2 + d3_fusion_factor_e1 + d3
        # d3_fusion = self.channelfusion_e2_d3(d3_fusion, d3)




        d2 = self.decoder2(d3)
        # d2_fusion = e1 + self.lay3_up_4(e3) + self.lay2_up_2(e2)
        # d2_fusion_factor = self.attention_2(d2_fusion, d2)
        # d2_fusion_factor_e1 = self.attention_2_minus_e1(e1, d2)
        # d2_fusion_factor_e3 = self.attention_2_minus_e3(self.lay3_up_4(e3), d2)
        # d2_fusion_factor_e2 = self.attention_2_minus_e2(self.lay2_up_2(e2), d2)
        d2_fusion_factor_e1 = self.channelfusion_e1_d2(e1, d2)
        d2_fusion_factor_e2 = self.channelfusion_e2_d2(self.lay2_up_2(e2), d2)
        d2_fusion_factor_e3 = self.channelfusion_e3_d2(self.lay3_up_4(e3), d2)
        d2 = torch.cat(((d2_fusion_factor_e1),
                        (d2_fusion_factor_e3),
                        (d2_fusion_factor_e2)), dim=1)
        # # d2 = torch.cat((d2, e1 * (d2_fusion_factor_e1),
        # #                 self.lay3_up_4(e3) * (d2_fusion_factor_e3),
        # #                 self.lay2_up_2(e2) * (d2_fusion_factor_e2)), dim=1)
        # #
        d2 = self.d2_reduction(d2)
        # d2 = d2_fusion_factor_e1 * d2_fusion_factor_e2 * d2_fusion_factor_e3
        # d2 = d2_fusion_factor_e1 + d2_fusion_factor_e2 + d2_fusion_factor_e3 + d2
        # d2_fusion = self.channelfusion_e1_d2(d2_fusion, d2)

        # d1_fusion = self.lay1_up_2(e1) + self.lay2_up_4(e2)
        d1 = self.decoder1(d2)
        # d1_fusion_factor = self.attention_1(d1_fusion, d1)
        # d1 = torch.cat((d1, self.lay1_up_2(e1) * d1_fusion_factor,
        #                 self.lay2_up_4(e2) * d1_fusion_factor), dim=1)
        # d1 = self.d1_reduction(d1)




        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DownFusionBlock(nn.Module):
    def __init__(self, in_channels, n_filters, stride_num=2):
        super(DownFusionBlock, self).__init__()

        self.stride_num = stride_num

        self.conv1 = nn.Conv2d(in_channels, n_filters//4, 1,bias=False)
        self.norm1 = nn.BatchNorm2d(n_filters//4)
        self.relu1 = nonlinearity

        self.conv2 = nn.Conv2d(n_filters//4, n_filters//4, 3, stride=self.stride_num, padding=1)
        self.norm2 = nn.BatchNorm2d(n_filters//4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(n_filters // 4, n_filters, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity



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

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x

class UpFusionBlock(nn.Module):
    def __init__(self, in_channels, n_filters, stride_num=2):
        super(UpFusionBlock, self).__init__()

        self.stride_num = stride_num

        if self.stride_num == 2:
            self.pad = 1
            self.k = 4
        elif self.stride_num == 4:
            self.pad = 0
            self.k =4
        elif self.stride_num ==8:
            self.pad = 0
            self.k = 8

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1,bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, self.k, stride=self.stride_num, padding=self.pad)
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

class GA(nn.Module):
    def __init__(self, channels_high, channels_low):
        super(GA, self).__init__()
        # Global Attention Upsample

        # self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        # self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)


        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        out = self.relu(fms_high_gp)

        return out



class DifPyramidNet_1(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(DifPyramidNet_1, self).__init__()

        filters = [64, 128, 256, 512] #ResNet34
        # filters = [64, 256, 512, 1024] #ResNet50
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        # self.lay1_down_2 = DownFusionBlock(filters[0], filters[1])
        # self.lay1_down_4 = DownFusionBlock(filters[0], filters[2])

        self.encoder2 = resnet.layer2
        # self.lay2_down_2 = DownFusionBlock(filters[1], filters[2])
        # self.lay2_up_2 = UpFusionBlock(filters[1], filters[0])

        self.encoder3 = resnet.layer3
        # self.lay3_up_2 = UpFusionBlock(filters[2], filters[1])
        # self.lay3_up_4 = UpFusionBlock(filters[2], filters[0], stride_num=4)

        self.dblock = DifPyramidBlock(256) #ResNet34
        # self.e3_block_up4 = UpFusionBlock(filters[2], filters[0], stride_num=4)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.Global_decoder3 = GA(filters[2], filters[1])
        # self.conv_de3 = nn.Conv2d(filters[1] * 4, filters[1], 1, 1, bias=False)
        # self.bn_conv_de3 = nn.BatchNorm2d(filters[1])

        self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.Global_decoder2 = GA(filters[1], filters[0])
        # self.conv_de2 = nn.Conv2d(filters[0]*5, filters[0], 1, 1, bias=False)
        # self.bn_conv_de2 = nn.BatchNorm2d(filters[0])


        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1,bias=False)

        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.finalrelu2 = nonlinearity
        # self.dropout = nn.Dropout(0.1)
        self.finalconv3 = nn.Conv2d(32, n_classes-1, 3, padding=1)

        self.d3_reduction = nn.Sequential(
            nn.Conv2d(filters[1] * 2 , filters[1], 1, 1, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )

        self.d2_reduction = nn.Sequential(
            nn.Conv2d(filters[0] * 3, filters[0], 1, 1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()



    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)


        # Decoder
        # d3_GA = self.Global_decoder3(e3_dblock)
        # d3 = torch.cat( (self.decoder3(e3_dblock), e2, self.lay1_down_2(e1),
        #                  self.lay3_up_2(e3)), dim=1)
        # d3 = nonlinearity(self.bn_conv_de3(self.conv_de3(d3)))
        d3 = self.decoder3(e3) + e2



        # d2_GA = self.Global_decoder2(d3)
        # d2 = torch.cat( (self.decoder2(d3),  e1, self.lay3_up_4(e3), self.lay2_up_2(e2),
        #                  self.e3_block_up4(e3_dblock)), dim=1)
        # d2 = nonlinearity(self.bn_conv_de2(self.conv_de2(d2)))
        # d3_fusion = torch.cat((d3, self.lay3_up_2(e3_dblock)), dim=1)
        # d3_fusion = self.d3_reduction(d3_fusion)
        # d2 = self.decoder2(d3_fusion) + e1
        d2 = self.decoder2(d3) + e1


        # d2_fusion = torch.cat((d2, self.lay3_up_4(e3_dblock), self.lay2_up_2(d3)), dim=1)
        # d2_fusion = self.d2_reduction(d2_fusion)
        # d1 = self.decoder1(d2_fusion)
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class DifPyramidNet_res(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(DifPyramidNet_res, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2

        self.block = nn.Sequential(
            nn.Conv2d(filters[1], filters[1]//4, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters[1]//4),
            nn.ReLU(True),

            nn.Conv2d(filters[1]//4, n_classes-1, 1)
        )


    def forward(self, x):
        # Encoder
        size = x.size()[2:]
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        out = self.block(e2)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)

        return F.sigmoid(out)


class DifPyramidNet(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(DifPyramidNet, self).__init__()

        filters = [64, 128, 256, 512] #ResNet34
        # filters = [64, 256, 512, 1024] #ResNet50
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder1_1 = Recurrent_block(filters[0])
        # self.encoder1_2 = Block(filters[0])
        # self.encoder1_1 = Block(filters[0])


        self.encoder2 = resnet.layer2
        self.encoder2_1 = Recurrent_block(filters[1])
        # self.encoder2_2 = Block(filters[1])
        # self.encoder2_1 = Block(filters[1])

        self.encoder3 = resnet.layer3

        self.dblock = DifPyramidBlock(256) #ResNet34


        self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.channelfusion_e2_d3 = ChannelAttention(filters[1]*2, filters[1])
        # self.SE_d3 = SEBlock(filters[1])
        self.SP_d3 = SPBlock(filters[1])
        # self.SP_d3 = AC_Attention(filters[1])
        # self.SP_d3 = SP_pooling(filters[1])
        # self.attention_3_minus_e2 = SpatialAttention(filters[1], filters[1], filters[1]//2)
        # self.cat_e2 = nn.Sequential(
        #     nn.Conv2d(filters[1]*2, filters[1], 1, 1, bias=False),
        #     nn.BatchNorm2d(filters[1]),
        #     nn.ReLU(inplace=True)
        # )

        self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.channelfusion_e1_d2 = ChannelAttention(filters[0] * 2 , filters[0])
        # self.SE_d2 = SEBlock(filters[0])
        self.SP_d2 = SPBlock(filters[0])
        # self.SP_d2 = AC_Attention(filters[0])
        # self.SP_d2 = SP_pooling(filters[0])
        # self.attention_2_minus_e1 = SpatialAttention(filters[0], filters[0], filters[0]//2)
        # self.cat_e1 = nn.Sequential(
        #     nn.Conv2d(filters[0] * 2, filters[0], 1, 1, bias=False),
        #     nn.BatchNorm2d(filters[0]),
        #     nn.ReLU(inplace=True)
        # )

        self.decoder1 = DecoderBlock(filters[0], filters[0])
        # self.SE_d1 = SEBlock(filters[0])
        self.SP_d1 = SPBlock(filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1,bias=False)

        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.finalrelu2 = nonlinearity
        # self.dropout = nn.Dropout(0.1)
        self.finalconv3 = nn.Conv2d(32, n_classes-1, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()



    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)


        # Decoder
        # d3 = self.decoder3(e3) + e2
        d3 = self.decoder3(e3)

        e2 = self.encoder2_1(e2)

        d3 = self.SP_d3(e2 + d3)


        # d2 = self.decoder2(d3) + e1
        d2 = self.decoder2(d3)

        e1 = self.encoder1_1(e1)

        d2 = self.SP_d2(e1 + d2)


        d1 = self.decoder1(d2)

        d1 = self.SP_d1(d1)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DifPyramidNet_base(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(DifPyramidNet_base, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.dblock = DifPyramidBlock(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes - 1, 3, padding=1)


    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class DifPyramidNet_less(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(DifPyramidNet_less, self).__init__()

        filters = [64, 128, 256, 512] #ResNet34
        # filters = [64, 256, 512, 1024] #ResNet50
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder1_1 = Recurrent_block(filters[0])

        self.encoder2 = resnet.layer2


        self.dblock = DifPyramidBlock(128) #ResNet34

        self.decoder2 = DecoderBlock(filters[1], filters[0])

        self.SP_d2 = SPBlock(filters[0])

        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.SP_d1 = SPBlock(filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1,bias=False)

        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.finalrelu2 = nonlinearity
        # self.dropout = nn.Dropout(0.1)
        self.finalconv3 = nn.Conv2d(32, n_classes-1, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()



    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)

        # Center
        e2 = self.dblock(e2)

        d2 = self.decoder2(e2)

        e1 = self.encoder1_1(e1)

        d2 = self.SP_d2(e1 + d2)


        d1 = self.decoder1(d2)

        d1 = self.SP_d1(d1)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class SEBlock(nn.Module):
    def __init__(self, channel):
        super(SEBlock, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(channel),
        #     nn.ReLU(inplace=True)
        # )
        self.Att = SELayer(channel, channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # out = self.conv(x)
        out = x * self.Att(x)
        return out

class SPBlock(nn.Module):
    def __init__(self, channel):
        super(SPBlock, self).__init__()

        # self.seblock = SELayer(channel, channel)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(channel, channel//4, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(channel//4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel//4, 1, 3, 1, 1, bias=False),
        #     nn.Sigmoid()
        #
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(channel, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()

        )

        # self.alpha = nn.Parameter(torch.zeros(1))
        # self.weights = nn.Parameter(torch.randn(2))
        # self.nums = 2
        # self._reset_parameters()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    # def _reset_parameters(self):
    #     init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        # out = self.seblock(x)
        # out = x * out
        master = self.conv1x1(x)
        out = self.conv(x)
        out = out * x
        # finalout = self.weights[0]*out + self.weights[1]*x
        # finalout = self.alpha * out + x
        finalout = out + master
        return finalout

class SP_pooling(nn.Module):
    def __init__(self, channel):
        super(SP_pooling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()

        )
        self.sigmod = nn.Sigmoid()

        self.weights = nn.Parameter(torch.randn(3))
        self.nums = 3
        self._reset_parameters()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)


    def forward(self, input):
        avg_out = torch.mean(input, dim=1, keepdim=True)
        avg_out = self.sigmod(avg_out) * input
        max_out, _ = torch.max(input, dim=1, keepdim=True)
        max_out = self.sigmod(max_out) * input
        out = self.conv(input)
        out = out * input


        # finalout = self.weights[0] * out + self.weights[1] * input
        finalout = self.weights[0]*avg_out + self.weights[1]*max_out + self.weights[2]*out
        return  finalout

class Block(nn.Module):
    def __init__(self, channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1, bias=False )
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Block_SE(nn.Module):
    def __init__(self, channel):
        super(Block_SE, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1, bias=False )
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.Att = SELayer(channel, channel)

        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        original_out = out
        out = self.Att(out)
        out = out * original_out

        out += residual
        out = self.relu(out)

        return out
class AC_Attention(nn.Module):
    def __init__(self, ch_out):
        super(AC_Attention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out//4),
            nn.ReLU(inplace=True)
        )

        self.squ_conv = nn.Sequential(
            nn.Conv2d(ch_out//4, ch_out//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out//4),

        )
        self.ver_conv = nn.Sequential(
            nn.Conv2d(ch_out//4, ch_out//4, kernel_size=(3 ,1), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(ch_out//4),

        )
        self.hor_conv = nn.Sequential(
            nn.Conv2d(ch_out//4, ch_out//4, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(ch_out//4),

        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(ch_out//4, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.sigmod = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.conv(x)
        out = self.squ_conv(out) + self.ver_conv(out) + self.hor_conv(out)
        out = nonlinearity(out)
        out = self.conv_1(out)
        out = x * out
        out = self.alpha*out + x

        return out

class AC_Attention_v2(nn.Module):
    def __init__(self, ch_out):
        super(AC_Attention_v2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out//4),
            nn.ReLU(inplace=True)
        )

        self.squ_conv = nn.Sequential(
            nn.Conv2d(ch_out//4, ch_out//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out//4, 1, 3, 1, 1, bias=False)

        )
        self.ver_conv = nn.Sequential(
            nn.Conv2d(ch_out//4, ch_out//4, kernel_size=(3 ,1), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(ch_out//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, 1, (3 ,1), 1, (1, 0), bias=False)

        )
        self.hor_conv = nn.Sequential(
            nn.Conv2d(ch_out//4, ch_out//4, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(ch_out//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, 1, (1, 3), 1, (0, 1), bias=False)

        )

        self.sigmod = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.conv(x)
        out = self.squ_conv(out) + self.ver_conv(out) + self.hor_conv(out)
        out = self.sigmod(out)

        out = x * out
        out = self.alpha*out + self.beta*x

        return out

class Square_Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Square_Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),

        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)


            x1 = self.conv(x + x1)

        return x1


class Ver_Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Ver_Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(ch_out),

        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)

        return x1


class Hor_Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Hor_Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(ch_out),

        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)

        return x1

class AC_Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(AC_Recurrent_block,self).__init__()
        self.squ = Square_Recurrent_block(ch_out, t=2)
        self.ver = Ver_Recurrent_block(ch_out, t=2)
        self.hor = Hor_Recurrent_block(ch_out, t=2)

    def forward(self, input):
        squ_out = self.squ(input)
        ver_out = self.ver(input)
        hor_out = self.hor(input)
        out = nonlinearity(squ_out + ver_out + hor_out)
        return  out

class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)


            x1 = self.conv(x + x1)

        return x1

class Recurrent_block_SE(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block_SE, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.Att = SELayer(ch_out, ch_out)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)


            x1 = self.conv(x + x1)

        original_out = x1
        out = self.Att(x1)
        out = out * original_out

        return out

class Recurrent_block_1(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block_1, self).__init__()
        self.t = t
        self.ch_out = ch_out
        # self.conv = nn.Sequential(
        #     nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(ch_out),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(ch_out)
        # )
        self.conv = Block(ch_out)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        for i in range(self.t):

            if i == 0:

                x1 = self.conv(x)

            x1 = self.conv(x + x1)

        return x1

class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DinkNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(model, nn.ConvTranspose2d):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class FusionLayer(nn.Module):
    def __init__(self, nums=7):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(nums))
        self.nums = nums
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        for i in range(self.nums):
            out = self.weights[i] * x[i] if i == 0 else out + self.weights[i] * x[i]
        return out

class RefineLayer(nn.Module):
    def __init__(self, nums=7):
        super(RefineLayer, self).__init__()
        self.num = nums
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.weights = nn.Parameter(torch.randn(nums))

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        l = [i.unsqueeze(1) for i in x]
        s = torch.cat(l, dim=1)
        batch_size, scale, channel, height, width = s.size()
        feat_a = s.view(batch_size, -1, channel * height * width)
        feat_a_transpose = s.view(batch_size, -1, channel * height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        # attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        # attention = self.softmax(attention_new)
        attention = self.softmax(attention)
        #
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, channel, height, width)
        out = self.beta * feat_e + s
        # out = feat_e + s
        finalout = 0
        for i in range(self.num):
            finalout = finalout + self.weights[i] * out[:, i, :, :, :]
        # finalout = out[:, 0, :, :, :] + out[:, 1, :, :, :]

        return finalout


class LayerEncoding(nn.Module):
    def __init__(self, channle = 1):
        super(LayerEncoding, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channle, channle, 3, 1 ,1, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(channle, channle, 3, 1, 1, bias=False)
        )

    def forward(self, input, x):
        input_sig = F.sigmoid(input)
        input = (1 + input_sig) * input
        fusion_x = 0
        for i in range(len(x)):
            weight = F.sigmoid(x[i])
            fusion_x = fusion_x + (1 - input_sig) * weight * x[i]
        out = input + fusion_x
        out = self.fc(out)

        return out


