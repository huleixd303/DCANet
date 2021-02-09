"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import torch.nn.functional as F
from ptsemseg.models.utils import *
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


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


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class SDDBlock(nn.Module):
    def __init__(self, channel):
        super(SDDBlock,self).__init__()
        # global pooling
        self.conv_gp = nn.Conv2d(channel, channel, 1, 1, 0, bias=False)
        self.bn_gp = nn.BatchNorm2d(channel)

        self.conv = nn.Conv2d(channel, channel, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(channel)

        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1, bias=False)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, bias=False)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4, bias=False)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8, bias=False)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16, bias=False)

        self.fc = nn.Linear(channel, channel // 8)
        self.fcs = nn.ModuleList([])
        for i in range(4):
            self.fcs.append(
                nn.Linear(channel // 8, channel)
            )
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        # x_master = self.conv(input)
        # x_master = self.bn(x_master)

        x_gp = nn.AvgPool2d(input.shape[2:])(input)
        x_gp = self.conv_gp(x_gp)
        x_gp = self.bn_gp(x_gp)

        #output of dilatecov
        dilate1_out = nonlinearity(self.bn(self.dilate1(input)))
        dilate2_out = nonlinearity(self.bn(self.dilate2((dilate1_out))))
        dilate3_out = nonlinearity(self.bn(self.dilate3(dilate2_out)))
        dilate4_out = nonlinearity(self.bn(self.dilate4(dilate3_out)))
        # dilate5_out = self.bn(self.dilate5(dilate4_out))

        #discriminative feature of dilatecov
        # out5_1 = nonlinearity(dilate5_out - dilate1_out)
        # out4_1 = nonlinearity(dilate4_out - dilate1_out)
        # out3_1 = nonlinearity(dilate3_out - dilate1_out)
        # out2_1 = nonlinearity(dilate1_out)
        #
        # out5_2 = nonlinearity(dilate5_out - dilate2_out)
        # out4_2 = nonlinearity(dilate4_out - dilate2_out)
        # out3_2 = nonlinearity(dilate2_out)
        #
        # out5_3 = nonlinearity(dilate5_out - dilate3_out)
        # out4_3 = nonlinearity(dilate3_out)
        #
        # out5_4 = nonlinearity(dilate4_out)

        #intergrating discriminative feature
        # sum_conv = out5_1 + out5_2 + out5_3 + out5_4 + \
        #     out4_1 + out4_2 + out4_3 + out3_1 + out3_2 \
        #     + out2_1
        sum_conv = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        # cat_conv = torch.cat([(out5_1).unsqueeze(1), (out4_1).unsqueeze(1), \
        #                      (out3_1).unsqueeze(1), (out2_1).unsqueeze(1), \
        #                      (out5_2).unsqueeze(1), (out4_2).unsqueeze(1), \
        #                      (out3_2).unsqueeze(1), (out5_3).unsqueeze(1), \
        #                      (out4_3).unsqueeze(1), (out5_4).unsqueeze(1)], 1)
        cat_conv = torch.cat([(dilate1_out).unsqueeze(1), (dilate2_out).unsqueeze(1),
                              (dilate3_out).unsqueeze(1), (dilate4_out).unsqueeze(1)], dim = 1)
        sum_conv_gp = nn.AvgPool2d(sum_conv.shape[2:])(sum_conv).squeeze(-1).squeeze(-1)
        feature_z = nonlinearity(self.fc(sum_conv_gp))
        for i, fc in enumerate(self.fcs):
            vector = fc(feature_z).unsqueeze(1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], 1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        out = (cat_conv * attention_vectors)
        out = out.sum(dim=1)

        #finalout = nonlinearity(x_gp + x_master * out)
        finalout = nonlinearity(x_gp + out)

        return finalout

class DifPyramidBlock(nn.Module):
    def __init__(self, channel):
        super(DifPyramidBlock,self).__init__()


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
        # self.dilate1 = LargeKernelConv(channel, (3, 3))
        # self.dilate2 = LargeKernelConv(channel, (5, 5))
        # self.dilate3 = LargeKernelConv(channel, (7, 7))
        # self.dilate4 = LargeKernelConv(channel, (9, 9))
        # self.dilate5 = LargeKernelConv(channel, (11, 11))

        # self.fc = nn.Linear(channel, channel // 8)
        # self.fcs = nn.ModuleList([])
        # for i in range(4):
        #     self.fcs.append(
        #         nn.Linear(channel // 8, channel)
        #     )
        # self.softmax = nn.Softmax(dim=1)

        # self.convadj = nn.Conv2d(channel*2, channel*2, 1, bias=False)
        # self.bn0 = nn.BatchNorm2d(channel*2)

        self.conv1 = nn.Conv2d(channel*4, channel, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)



        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        x_master = self.conv(input)
        x_master = self.bn(x_master)

        x_gp = nn.AvgPool2d(input.shape[2:])(input)
        x_gp = self.conv_gp(x_gp)
        x_gp = self.bn_gp(x_gp)

        dilate1_out = self.bn(self.dilate1(input))
        # dilate1_out_relu = nonlinearity(dilate1_out)
        dilate2_out = self.bn(self.dilate2((dilate1_out)))
        # dilate2_out_relu = nonlinearity(dilate2_out)
        dilate3_out = self.bn(self.dilate3((dilate2_out)))
        # dilate3_out_relu = nonlinearity(dilate3_out)
        dilate4_out = self.bn(self.dilate4((dilate3_out)))
        # dilate4_out_relu = nonlinearity(dilate4_out)
        dilate5_out = self.bn(self.dilate5((dilate4_out)))
        #dilate5_out = nonlinearity(dilate5_out)

        out1 = nonlinearity(dilate2_out- dilate1_out)
        out2 = nonlinearity(dilate3_out - dilate2_out)
        out3 = nonlinearity(dilate4_out - dilate3_out)
        out4 = nonlinearity(dilate5_out - dilate4_out)

        #intergrating discriminative feature
        # out = out1 + out2 + out3 + out4
        out = torch.cat((out1, out2, out3, out4), dim=1)

        x = nonlinearity(self.bn1(self.conv1(out)))

        # out_adj_out1_2 = torch.cat((out1, out2), dim=1)
        # out_adj_out1_2 = nonlinearity(self.bn0(self.convadj(out_adj_out1_2)))
        #
        # out_adj_out3_4 = torch.cat((out3, out4), dim=1)
        # out_adj_out3_4 = nonlinearity(self.bn0(self.convadj(out_adj_out3_4)))
        #
        # out = torch.cat((out_adj_out1_2, out_adj_out3_4), dim=1)
        # x = nonlinearity(self.bn1(self.conv1(out)))


        # cat_conv = torch.cat([(out1).unsqueeze(1), (out2).unsqueeze(1), \
        #                      (out3).unsqueeze(1), (out4).unsqueeze(1), \
        #                      ], 1)
        # sum_conv_gp = nn.AvgPool2d(sum_conv.shape[2:])(sum_conv).squeeze(-1).squeeze(-1)
        # feature_z = nonlinearity(self.fc(sum_conv_gp))
        # for i, fc in enumerate(self.fcs):
        #     vector = fc(feature_z).unsqueeze(1)
        #     if i == 0:
        #         attention_vectors = vector
        #     else:
        #         attention_vectors = torch.cat([attention_vectors, vector], 1)
        #
        # attention_vectors = self.softmax(attention_vectors)
        # attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        # out = (cat_conv * attention_vectors)
        # out = out.sum(dim=1)
        #out = out1 + out2 + out3 + out4
        #out = F.sigmoid(out)

        finalout = nonlinearity(x_master*x + x_gp)

        return finalout



class upsample(nn.Module):
    def __init__(self, in_channels):
        super(upsample, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 8, 1)

        self.relu1 = nonlinearity

        # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1)
        # self.norm2 = nn.BatchNorm2d(in_channels // 4)
        # self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(8, 16, 1)

        self.relu3 = nonlinearity

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x, factor):
        x = self.conv1(x)

        x = self.relu1(x)

        x = F.interpolate(x, scale_factor=2.0**factor, mode='bilinear', align_corners=True)

        x = self.conv3(x)

        x = self.relu3(x)
        return x


class shuffleblock(nn.Module):
    def __init__(self, low_channel, mid_channel, high_channel):
        super(shuffleblock, self).__init__()

        self.shuff2high = nn.Sequential(
            nn.Conv2d(low_channel, high_channel, 1, 1, bias=False),
            nn.BatchNorm2d(high_channel),
            nn.ReLU(inplace=True)
        )
        self.shuff2mid = nn.Sequential(
            nn.Conv2d(low_channel, mid_channel, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        self.shuff2low = nn.Sequential(
            nn.Conv2d(low_channel, low_channel, 1, 1, bias=False),
            nn.BatchNorm2d(low_channel),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, groups):
        a, b, c = channel_shuffle(input, groups)
        a = self.shuff2low(a)
        b = self.shuff2mid(b)
        c = self.shuff2high(c)

        return a, b, c



def channel_shuffle(x, groups):
    concat_fm = torch.cat(x, dim = 1)
    batchsize, num_channel, height, width = concat_fm.size()
    channel_per_group = num_channel//groups
    concat_fm = concat_fm.view(batchsize, groups, channel_per_group, height, width)
    concat_fm = torch.transpose(concat_fm, 1, 2).contiguous()
    concat_fm = concat_fm.view(batchsize, -1, height, width)
    a, b, c = concat_fm.split(64, dim = 1)
    b = F.interpolate(b, size = [64, 64], mode = "bilinear")
    c = F.interpolate(c, size = [32, 32], mode = "bilinear")

    return a, b, c




class SDDlinknet_1(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(SDDlinknet_1, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1

        self.encoder2 = resnet.layer2
        # self.e2upsample = upsample(filters[1], 2)

        self.encoder3 = resnet.layer3

        self.dblock = DifPyramidBlock(256)
        # self.e3upsample = upsample(filters[2], 4)

        # self.shuffleblock = shuffleblock(filters[0], filters[1], filters[2])

        self.outconv_e3 = conv2DBatchNormRelu(
            in_channels=256, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )
        # self.outconv_e3_b2u = conv2DBatchNormRelu(
        #     in_channels=256, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        # )

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.outconv_d3 = conv2DBatchNormRelu(
            in_channels=128, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )
        # self.outconv_d3_b2u = conv2DBatchNormRelu(
        #     in_channels=128, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        # )
        self.convcat3 = nn.Conv2d(2, 1, 1, bias=False)

        self.decoder3 = DecoderBlock(filters[2], filters[1])


        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.outconv_d2 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )
        # self.outconv_d2_b2u = conv2DBatchNormRelu(
        #     in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        # )
        self.convcat2 = nn.Conv2d(3, 1, 1, bias=False)


        # self.decoder1 = DecoderBlock(filters[0], filters[0])
        # self.outconv_d1 = conv2DBatchNormRelu(
        #     in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        # )
        # self.outconv_d1_b2u = conv2DBatchNormRelu(
        #     in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        # )
        # self.convcat1 = nn.Conv2d(4, 1, 1, bias=False)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 8, 4, 2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes-1, 3, padding=1)
        self.convcat0 = nn.Conv2d(4, 1, 1, bias=False)
        self.fuse = FusionLayer()

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)

        e2 = self.encoder2(e1)
        # s_e2 = self.e2upsample(e2)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)
        # s_e3 = self.e3upsample(e3)

        # fm = [e1, s_e2, s_e3]
        # s_e1, s_e2, s_e3 = self.shuffleblock(fm, groups = 3)
        # s_e3 = self.dblock(e3)

        e3_aux_m_32 = self.outconv_e3(e3)
        e3_aux_m_512 = F.interpolate(e3_aux_m_32, scale_factor=16.0, mode='bilinear', align_corners=True)
        e3_aux_b2u = e3_aux_m_512
        e3_aux_t2d = e3_aux_m_512
        e3_aux_t2d_sig = F.sigmoid(e3_aux_t2d)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d3_aux_m = self.outconv_d3(d3)
        d3_aux_m = F.interpolate(d3_aux_m, scale_factor=8.0, mode='bilinear', align_corners=True)
        d3_aux_b2u = d3_aux_m
        d3_aux_t2d = d3_aux_m
        d3_aux_t2d = torch.cat((d3_aux_t2d, e3_aux_t2d), dim=1)
        d3_aux_t2d = self.convcat3(d3_aux_t2d)
        d3_aux_t2d_sig = F.sigmoid(d3_aux_t2d)


        d2 = self.decoder2(d3) + e1
        d2_aux_m = self.outconv_d2(d2)
        d2_aux_m = F.interpolate(d2_aux_m, scale_factor=4.0, mode='bilinear', align_corners=True)
        d2_aux_b2u = d2_aux_m
        d2_aux_t2d = d2_aux_m
        d2_aux_t2d = torch.cat((e3_aux_t2d, d3_aux_t2d, d2_aux_t2d), dim=1)
        d2_aux_t2d = self.convcat2(d2_aux_t2d)
        d2_aux_t2d_sig = F.sigmoid(d2_aux_t2d)


        # d1 = self.decoder1(d2)
        # d1_aux_m = self.outconv_d1(d1)
        # d1_aux_m = F.interpolate(d1_aux_m, scale_factor=2.0, mode='bilinear', align_corners=True)
        # d1_aux_b2u = F.interpolate(self.outconv_d1_b2u(d1), scale_factor=2.0, mode='bilinear', align_corners=True)
        # d1_aux_t2d = d1_aux_m
        # d1_aux_t2d = torch.cat((e3_aux_t2d, d3_aux_t2d, d2_aux_t2d, d1_aux_t2d), dim=1)
        # d1_aux_t2d = self.convcat1(d1_aux_t2d)
        # d1_aux_t2d_sig = F.sigmoid(d1_aux_t2d)


        # Final Classification
        out = self.finaldeconv1(d2)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        out_b2u = out
        out_t2d = out

        out_b2u_sig = F.sigmoid(out_b2u)
        out_t2d = torch.cat((e3_aux_t2d, d3_aux_t2d, d2_aux_t2d, out_t2d), dim=1)
        out_t2d = self.convcat0(out_t2d)
        out_t2d_sig = F.sigmoid(out_t2d)

        # d1_aux_b2u = torch.cat((out_b2u, d1_aux_b2u), dim=1)
        # d1_aux_b2u = self.convcat3(d1_aux_b2u)
        # d1_aux_b2u_sig = F.sigmoid(d1_aux_b2u)

        d2_aux_b2u = torch.cat((out_b2u, d2_aux_b2u), dim=1)
        d2_aux_b2u = self.convcat3(d2_aux_b2u)
        d2_aux_b2u_sig = F.sigmoid(d2_aux_b2u)

        d3_aux_b2u = torch.cat((out_b2u, d2_aux_b2u, d3_aux_b2u), dim=1)
        d3_aux_b2u = self.convcat2(d3_aux_b2u)
        d3_aux_b2u_sig = F.sigmoid(d3_aux_b2u)

        e3_aux_b2u = torch.cat((out_b2u, d2_aux_b2u, d3_aux_b2u, e3_aux_b2u), dim=1)
        e3_aux_b2u = self.convcat0(e3_aux_b2u)
        e3_aux_b2u_sig = F.sigmoid(e3_aux_b2u)

        outfusion = [out_b2u, d2_aux_b2u, d3_aux_b2u, e3_aux_b2u,
                     out_t2d]
        finalrefuse = self.fuse(outfusion)
        finalrefuse = F.sigmoid(finalrefuse)
        if self.training:
            return e3_aux_b2u_sig, d3_aux_b2u_sig, d2_aux_b2u_sig, out_b2u_sig, e3_aux_t2d_sig, d3_aux_t2d_sig, \
                   d2_aux_t2d_sig, out_t2d_sig, finalrefuse
        else:  # eval mode
            return finalrefuse
        # return F.sigmoid(out)
        # return out


class SDDlinknet_2(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(SDDlinknet_2, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1

        self.encoder2 = resnet.layer2
        # self.e2upsample = upsample(filters[1], 2)

        self.encoder3 = resnet.layer3

        self.dblock = DifPyramidBlock(256)
        # self.e3upsample = upsample(filters[2], 4)

        # self.shuffleblock = shuffleblock(filters[0], filters[1], filters[2])

        self.outconv_e3 = conv2DBatchNormRelu(
            in_channels=256, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )


        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.outconv_d3 = conv2DBatchNormRelu(
            in_channels=128, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )

        self.convcat3 = nn.Conv2d(2, 1, 1, bias=False)

        self.decoder3 = DecoderBlock(filters[2], filters[1])


        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.outconv_d2 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        )

        self.convcat2 = nn.Conv2d(3, 1, 1, bias=False)
        self.downsample_2_d3 = nn.Conv2d(1, 1, 3, 2, 1, bias=False)
        self.downsample_2_d2 = nn.Conv2d(1, 1, 3, 2, 1, bias=False)
        self.downsample_4_d2 = nn.Conv2d(1, 1, 3, 4, 1, bias=False)
        self.downsample_4_out = nn.Conv2d(1, 1, 3, 4, 1, bias=False)
        self.downsample_8_out = nn.Conv2d(1, 1, 3, 8, 1, bias=False)
        self.downsample_16_out = nn.Conv2d(1, 1, 3, 16, 1, bias=False)


        # self.upsample_e3 = upsample(32)
        # self.upsample_d3 = upsample(32)
        # self.upsample_d2 = upsample(32)


        # self.decoder1 = DecoderBlock(filters[0], filters[0])
        # self.outconv_d1 = conv2DBatchNormRelu(
        #     in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        # )
        # self.outconv_d1_b2u = conv2DBatchNormRelu(
        #     in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        # )
        # self.convcat1 = nn.Conv2d(4, 1, 1, bias=False)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 8, 4, 2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes-1, 3, padding=1)
        self.convcat0 = nn.Conv2d(4, 1, 1, bias=False)
        self.fuse = FusionLayer()
        self.convcat = conv2DBatchNormRelu(128, 32, 3, 1, 1, bias=False)
        self.conv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)


    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)

        e2 = self.encoder2(e1)
        # s_e2 = self.e2upsample(e2)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)
        # s_e3 = self.e3upsample(e3)

        # fm = [e1, s_e2, s_e3]
        # s_e1, s_e2, s_e3 = self.shuffleblock(fm, groups = 3)
        # s_e3 = self.dblock(e3)

        # e3_aux_m_32 = self.outconv_e3(e3)
        # e3_aux_m_512 = self.upsample_e3(e3_aux_m_32, 4)
        # e3_aux_m_64 = self.upsample_e3(e3_aux_m_32, 1)
        # e3_aux_m_128 = self.upsample_e3(e3_aux_m_32, 2)
        # e3_aux_m_512 = F.interpolate(e3_aux_m_32, scale_factor=16.0, mode='bilinear', align_corners=True)
        # e3_aux_m_512 = nonlinearity(self.conv(e3_aux_m_512))
        # e3_aux_m_64 = F.interpolate(e3_aux_m_32, scale_factor=2.0, mode='bilinear', align_corners=True)
        # e3_aux_m_64 = nonlinearity(self.conv(e3_aux_m_64))
        # e3_aux_m_128 = F.interpolate(e3_aux_m_32, scale_factor=4.0, mode='bilinear', align_corners=True)
        # e3_aux_m_128 = nonlinearity(self.conv(e3_aux_m_128))

        # Decoder
        d3 = self.decoder3(e3) + e2
        # d3_aux_m_64 = self.outconv_d3(d3)
        # d3_aux_m_512 = self.upsample_e3(d3_aux_m_64, 3)
        # d3_aux_m_128 = self.upsample_e3(d3_aux_m_64, 1)
        # d3_aux_m_512 = F.interpolate(d3_aux_m_64, scale_factor=8.0, mode='bilinear', align_corners=True)
        # d3_aux_m_512 = nonlinearity(self.conv(d3_aux_m_512))
        # d3_aux_m_128 = F.interpolate(d3_aux_m_64, scale_factor=2.0, mode='bilinear', align_corners=True)
        # d3_aux_m_128 = nonlinearity(self.conv(d3_aux_m_128))
        # d3_aux_m_32 = self.downsample_2_d3(d3_aux_m_64)


        d2 = self.decoder2(d3) + e1
        # d2_aux_m_128 = self.outconv_d2(d2)
        # d2_aux_m_512 = self.upsample_e3(d2_aux_m_128, 2)
        # d2_aux_m_512 = F.interpolate(d2_aux_m_128, scale_factor=4.0, mode='bilinear', align_corners=True)
        # d2_aux_m_512 = nonlinearity(self.conv(d2_aux_m_512))
        # d2_aux_m_64 = self.downsample_2_d2(d2_aux_m_128)
        # d2_aux_m_32 = self.downsample_4_d2(d2_aux_m_128)


        # d1 = self.decoder1(d2)
        # d1_aux_m = self.outconv_d1(d1)
        # d1_aux_m = F.interpolate(d1_aux_m, scale_factor=2.0, mode='bilinear', align_corners=True)
        # d1_aux_b2u = F.interpolate(self.outconv_d1_b2u(d1), scale_factor=2.0, mode='bilinear', align_corners=True)
        # d1_aux_t2d = d1_aux_m
        # d1_aux_t2d = torch.cat((e3_aux_t2d, d3_aux_t2d, d2_aux_t2d, d1_aux_t2d), dim=1)
        # d1_aux_t2d = self.convcat1(d1_aux_t2d)
        # d1_aux_t2d_sig = F.sigmoid(d1_aux_t2d)


        # Final Classification
        out_512 = self.finaldeconv1(d2)
        out_512 = self.finalrelu1(out_512)
        out_512 = self.finalconv2(out_512)
        out_512 = self.finalrelu2(out_512)
        out_512 = self.finalconv3(out_512)


        out_128 = self.downsample_4_out(out_512)
        out_64 = self.downsample_8_out(out_512)
        out_32 = self.downsample_16_out(out_512)

        out_m = torch.cat((out_512, e3_aux_m_512, d3_aux_m_512, d2_aux_m_512), dim=1)
        d2_aux_m = torch.cat((out_128, d2_aux_m_128, d3_aux_m_128, e3_aux_m_128), dim=1)
        d3_aux_m = torch.cat((out_64, d2_aux_m_64, d3_aux_m_64, e3_aux_m_64), dim=1)
        e3_aux_m = torch.cat((out_32, d2_aux_m_32, d3_aux_m_32, e3_aux_m_32), dim=1)

        out_m = self.convcat0(out_m)
        out_m = self.conv(out_m)
        out_m = F.sigmoid(out_m)

        d2_aux_m = self.convcat0(d2_aux_m)
        d2_aux_m = self.conv(d2_aux_m)
        d2_aux_m = F.sigmoid(d2_aux_m)

        d3_aux_m = self.convcat0(d3_aux_m)
        d3_aux_m = self.conv(d3_aux_m)
        d3_aux_m = F.sigmoid(d3_aux_m)

        e3_aux_m = self.convcat0(e3_aux_m)
        e3_aux_m = self.conv(e3_aux_m)
        e3_aux_m = F.sigmoid(e3_aux_m)


        if self.training:
            return e3_aux_m, d3_aux_m, d2_aux_m, out_m
        else:  # eval mode
            return out_m


class SDDlinknet(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(SDDlinknet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1

        self.encoder2 = resnet.layer2
        # self.e2upsample = upsample(filters[1], 2)

        self.encoder3 = resnet.layer3

        self.dblock = DifPyramidBlock(256)
        # self.e3upsample = upsample(filters[2], 4)

        # self.shuffleblock = shuffleblock(filters[0], filters[1], filters[2])

        self.outconv_e3 = conv2DBatchNormRelu(
            in_channels=256, k_size=3, n_filters=16, padding=1, stride=1, bias=False
        )


        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.outconv_d3 = conv2DBatchNormRelu(
            in_channels=128, k_size=3, n_filters=16, padding=1, stride=1, bias=False
        )

        self.convcat3 = nn.Conv2d(2, 1, 1, bias=False)

        self.decoder3 = DecoderBlock(filters[2], filters[1])


        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.outconv_d2 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=16, padding=1, stride=1, bias=False
        )

        self.convcat2 = nn.Conv2d(3, 1, 1, bias=False)
        self.downsample_2_d3 = nn.Conv2d(16, 16, 3, 2, 1, bias=False)
        self.downsample_2_d2 = nn.Conv2d(16, 16, 3, 2, 1, bias=False)
        self.downsample_4_d2 = nn.Conv2d(16, 16, 3, 4, 1, bias=False)
        self.downsample_4_out = nn.Conv2d(16, 16, 3, 4, 1, bias=False)
        self.downsample_8_out = nn.Conv2d(16, 16, 3, 8, 1, bias=False)
        self.downsample_16_out = nn.Conv2d(16, 16, 3, 16, 1, bias=False)


        self.upsample_e3 = upsample(16)
        self.upsample_d3 = upsample(16)
        self.upsample_d2 = upsample(16)


        # self.decoder1 = DecoderBlock(filters[0], filters[0])
        # self.outconv_d1 = conv2DBatchNormRelu(
        #     in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        # )
        # self.outconv_d1_b2u = conv2DBatchNormRelu(
        #     in_channels=64, k_size=3, n_filters=1, padding=1, stride=1, bias=False
        # )
        # self.convcat1 = nn.Conv2d(4, 1, 1, bias=False)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 16, 8, 4, 2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(16, n_classes-1, 3, padding=1)
        self.convcat0 = nn.Conv2d(4, 1, 1, bias=False)
        self.fuse = FusionLayer()
        self.convcat = conv2DBatchNormRelu(64, 16, 3, 1, 1, bias=False)
        self.conv = nn.Conv2d(16, 16, 3, 1, 1, bias=False)


    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)

        e2 = self.encoder2(e1)
        # s_e2 = self.e2upsample(e2)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)
        # s_e3 = self.e3upsample(e3)

        # fm = [e1, s_e2, s_e3]
        # s_e1, s_e2, s_e3 = self.shuffleblock(fm, groups = 3)
        # s_e3 = self.dblock(e3)

        e3_aux_m_32 = self.outconv_e3(e3)
        e3_aux_m_512 = self.upsample_e3(e3_aux_m_32, 4)
        e3_aux_m_64 = self.upsample_e3(e3_aux_m_32, 1)
        e3_aux_m_128 = self.upsample_e3(e3_aux_m_32, 2)
        # e3_aux_m_512 = F.interpolate(e3_aux_m_32, scale_factor=16.0, mode='bilinear', align_corners=True)
        # e3_aux_m_512 = self.conv(e3_aux_m_512)
        # e3_aux_m_64 = F.interpolate(e3_aux_m_32, scale_factor=2.0, mode='bilinear', align_corners=True)
        # e3_aux_m_64 = self.conv(e3_aux_m_64)
        # e3_aux_m_128 = F.interpolate(e3_aux_m_32, scale_factor=4.0, mode='bilinear', align_corners=True)
        # e3_aux_m_126 = self.conv(e3_aux_m_128)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d3_aux_m_64 = self.outconv_d3(d3)
        d3_aux_m_512 = self.upsample_e3(d3_aux_m_64, 3)
        d3_aux_m_128 = self.upsample_e3(d3_aux_m_64, 1)
        # d3_aux_m_512 = F.interpolate(d3_aux_m_64, scale_factor=8.0, mode='bilinear', align_corners=True)
        # d3_aux_m_512 = self.conv(d3_aux_m_512)
        # d3_aux_m_128 = F.interpolate(d3_aux_m_64, scale_factor=2.0, mode='bilinear', align_corners=True)
        # d3_aux_m_128 = self.conv(d3_aux_m_128)
        d3_aux_m_32 = self.downsample_2_d3(d3_aux_m_64)


        d2 = self.decoder2(d3) + e1
        d2_aux_m_128 = self.outconv_d2(d2)
        d2_aux_m_512 = self.upsample_e3(d2_aux_m_128, 2)
        # d2_aux_m_512 = F.interpolate(d2_aux_m_128, scale_factor=4.0, mode='bilinear', align_corners=True)
        # d2_aux_m_512 = self.conv(d2_aux_m_512)
        d2_aux_m_64 = self.downsample_2_d2(d2_aux_m_128)
        d2_aux_m_32 = self.downsample_4_d2(d2_aux_m_128)


        # d1 = self.decoder1(d2)
        # d1_aux_m = self.outconv_d1(d1)
        # d1_aux_m = F.interpolate(d1_aux_m, scale_factor=2.0, mode='bilinear', align_corners=True)
        # d1_aux_b2u = F.interpolate(self.outconv_d1_b2u(d1), scale_factor=2.0, mode='bilinear', align_corners=True)
        # d1_aux_t2d = d1_aux_m
        # d1_aux_t2d = torch.cat((e3_aux_t2d, d3_aux_t2d, d2_aux_t2d, d1_aux_t2d), dim=1)
        # d1_aux_t2d = self.convcat1(d1_aux_t2d)
        # d1_aux_t2d_sig = F.sigmoid(d1_aux_t2d)


        # Final Classification
        out_512 = self.finaldeconv1(d2)
        out_512 = self.finalrelu1(out_512)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        # out = self.finalconv3(out)


        out_128 = self.downsample_4_out(out_512)
        out_64 = self.downsample_8_out(out_512)
        out_32 = self.downsample_16_out(out_512)

        out_m = torch.cat((out_512, e3_aux_m_512, d3_aux_m_512, d2_aux_m_512), dim=1)
        d2_aux_m = torch.cat((out_128, d2_aux_m_128, d3_aux_m_128, e3_aux_m_128), dim=1)
        d3_aux_m = torch.cat((out_64, d2_aux_m_64, d3_aux_m_64, e3_aux_m_64), dim=1)
        e3_aux_m = torch.cat((out_32, d2_aux_m_32, d3_aux_m_32, e3_aux_m_32), dim=1)

        out_m = self.convcat(out_m)
        out_m = self.finalconv3(out_m)
        out_m = F.sigmoid(out_m)

        d2_aux_m = self.convcat(d2_aux_m)
        d2_aux_m = self.finalconv3(d2_aux_m)
        d2_aux_m = F.sigmoid(d2_aux_m)

        d3_aux_m = self.convcat(d3_aux_m)
        d3_aux_m = self.finalconv3(d3_aux_m)
        d3_aux_m = F.sigmoid(d3_aux_m)

        e3_aux_m = self.convcat(e3_aux_m)
        e3_aux_m = self.finalconv3(e3_aux_m)
        e3_aux_m = F.sigmoid(e3_aux_m)


        if self.training:
            return e3_aux_m, d3_aux_m, d2_aux_m, out_m
        else:  # eval mode
            return out_m



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


class FusionLayer(nn.Module):
    def __init__(self, nums=5):
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