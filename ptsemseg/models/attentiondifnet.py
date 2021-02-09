"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)

class SELayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, reduction = 2):
        super(ChannelAttention, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        #fm = nonlinearity(self.bn(x1 + x2))
        channel_attetion = self.channel_attention(fm)
        fm = x1 * channel_attetion + x2
        # fm = x1*channel_attetion
        #fm = nonlinearity(self.bn(fm))

        return fm

class SpatialAttention(nn.Module):
    def __init__(self,channel_low, channel_high, channel_mid):
        super(SpatialAttention,self).__init__()

        self.h_m = nn.Sequential(
            nn.Conv2d(channel_high, channel_mid, 1, 1, bias=False),
            nn.BatchNorm2d(channel_mid)
        )

        self.l_m = nn.Sequential(
            nn.Conv2d(channel_low, channel_mid, 1, 1, bias=False),
            nn.BatchNorm2d(channel_mid)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(channel_mid, 1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.convblock = nn.Sequential(
            nn.Conv2d(channel_high*2, channel_low, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel_low),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_low, channel_low, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel_low),
            nn.ReLU(inplace=True)
        )
        self.SKnum = SKBlock(channel_mid)
    def forward(self, lowfm, highfm):
        h_m = self.h_m(highfm)
        l_m = self.l_m(lowfm)
        #fms = nonlinearity(self.SKnum(h_m, l_m))
        fms = nonlinearity(h_m + l_m)
        psi = self.psi(fms) * lowfm
        out = torch.cat((psi, highfm), 1)
        finalout = self.convblock(out)

        return finalout


class GAU(nn.Module):
    def __init__(self, channels_high, channels_low):
        super(GAU, self).__init__()
        # Global Attention Upsample
        # self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)
        self.decoder = DecoderBlock(channels_high, channels_low)

        # if upsample:
        #     self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
        #     self.bn_upsample = nn.BatchNorm2d(channels_low)
        # else:
        #     self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        #     self.bn_reduction = nn.BatchNorm2d(channels_low)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
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
        #fms_high_gp = nonlinearity(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        # if self.upsample:
        #     out = self.relu(
        #         self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        # else:
        #     out = self.relu(
        #         self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        out = nonlinearity(self.decoder(fms_high) + fms_att)

        return out

class FA(nn.Module):
    def __init__(self, channel_high, channel_low):
        super(FA, self).__init__()
        self.decoder = DecoderBlock(channel_high, channel_low)

        self.conv1x1 = nn.Conv2d(channel_high, channel_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channel_low)

        self.conv3x3 = nn.Conv2d(channel_low, channel_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channel_low)

    def forward(self, fms_low, fms_high):
        fms_low_branch = (self.bn_low(self.conv3x3(fms_low)))

        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)

        fms_high_branch = self.decoder(fms_high)
        #fms_high_branch = self.bn_high(self.conv1x1(fms_high_branch))

        out = nonlinearity(fms_high_gp + fms_high_branch*fms_low_branch)

        return out


class Conv2(nn.Module):
    def __init__(self, channel):
        super(Conv2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs



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

class SKBlock(nn.Module):
    def __init__(self,channel):
        super(SKBlock,self).__init__()
        self.bn = nn.BatchNorm2d(channel)
        self.fc = nn.Linear(channel, 32)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                nn.Linear(32, channel)
            )
        self.softmax = nn.Softmax(dim=1)
        self.relu = nonlinearity

    def forward(self, fms_low, fms_high):
        sum_fms = fms_low + fms_high
        cat_fms = torch.cat([fms_low.unsqueeze(1), fms_high.unsqueeze(1)], 1)
        sum_conv_gp = nn.AvgPool2d(sum_fms.shape[2:])(sum_fms).squeeze_()
        feature_z = nonlinearity(self.fc(sum_conv_gp))
        for i, fc in enumerate(self.fcs):
            vector = fc(feature_z).unsqueeze(1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], 1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        out = (cat_fms * attention_vectors)
        finalout = out.sum(dim=1)

        return finalout


class PyramidBlock(nn.Module):
    def __init__(self, channel):
        super(PyramidBlock, self).__init__()


        #global pooling
        #self.conv_gp = nn.Conv2d(channel, channel, 1, 1, 0)
        #self.bn_gp = nn.BatchNorm2d(channel)

        #pyamidblock
        for i in range(1, 4):
            self.add_module(
                "respool3_{}".format(i),
                nn.Sequential(
                    nn.Conv2d(channel, channel, 3, 1, 1, groups=32, bias=False),
                    nn.BatchNorm2d(channel),
                    #nn.MaxPool2d(3, 1, 1)
                ))
            self.add_module(
                "respool5_{}".format(i),
                nn.Sequential(
                    nn.Conv2d(channel, channel, 5, 1, 2, groups=32, bias=False),
                    nn.BatchNorm2d(channel),
                    #nn.MaxPool2d(5, 1, 2)
                )
            )
            self.add_module(
                "respool7_{}".format(i),
                nn.Sequential(
                    nn.Conv2d(channel, channel, 7, 1, 3, groups=32, bias=False),
                    nn.BatchNorm2d(channel),
                    #nn.MaxPool2d(7, 1, 3)
                )
            )
            # self.add_module(
            #   "respool9_{}".format(i),
            #   nn.Sequential(
            #        nn.Conv2d(channel, channel, 9, 1, 4,groups=32,bias=False),
            #        nn.BatchNorm2d(channel),
            #        #nn.MaxPool2d(9, 1, 4)
            #    )
            # )

        self.bn = nn.BatchNorm2d(channel)
        self.fc = nn.Linear(channel, channel // 8)
        self.fcs = nn.ModuleList([])
        for i in range(3):
            self.fcs.append(
                nn.Linear(channel // 8, channel)
            )
        self.softmax = nn.Softmax(dim=1)
        self.relu = nonlinearity



    def forward(self, x):
        #input_gp = nn.AvgPool2d(x.shape[2 : ])(x)
        #input_gp = self.conv_gp(input_gp)
        #input_gp = self.bn_gp(input_gp)

        x3, x5, x7= x, x, x
        respool3path, respool5path, respool7path = x3, x5, x7

        for i in range(1, 4):
            residual3 = respool3path
            respool3path = self.__getattr__("respool3_{}".format(i))(respool3path)
            respool3path = self.relu(residual3 + respool3path)
            #x3 = self.relu(x3)

            residual5 = respool5path
            respool5path = self.__getattr__("respool5_{}".format(i))(respool5path)
            respool5path = self.relu(residual5 + respool5path)
            #x5 = self.relu(x5)

            residual7 = respool7path
            respool7path = self.__getattr__("respool7_{}".format(i))(respool7path)
            respool7path = self.relu(residual7 + respool7path)
            #x7 = self.relu(x7)

            # residual9 = respool9path
            # respool9path = self.__getattr__("respool9_{}".format(i))(respool9path)
            # respool9path = self.relu(residual9 + respool9path)
            # #x9 = self.relu(x9)

        sum_conv = respool3path + respool5path + respool7path
        cat_conv = torch.cat([respool3path.unsqueeze(1), respool5path.unsqueeze(1), \
                              respool7path.unsqueeze(1)], 1)
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
        finalout = out.sum(dim=1)



        return finalout




class DifPyramidBlock(nn.Module):
    def __init__(self, channel):
        super(DifPyramidBlock,self).__init__()
        #global pooling
        self.conv_gp = nn.Conv2d(channel, channel, 1, 1, 0)
        self.bn_gp = nn.BatchNorm2d(channel)

        self.conv = nn.Conv2d(channel, channel, 1 , 1)
        self.bn = nn.BatchNorm2d(channel)

        self.dilate0 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.channelExtension = nn.Conv2d(channel*4, channel, kernel_size=1, dilation=1, padding=0)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        #self.bnreduction = nn.BatchNorm2d(channel//4)
        #self.ChannelAttention = ChannelAttention(channel*2, channel, 4)

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

        # x = self.dilate0(input)
        # x = self.bnreduction(x)
        # x = nonlinearity(x)
        dilate1_out = self.bn(self.dilate1(input))
        #dilate1_out = nonlinearity(dilate1_out)
        dilate2_out = self.bn(self.dilate2(nonlinearity(dilate1_out)))
        #dilate2_out = nonlinearity(dilate2_out)
        dilate3_out = self.bn(self.dilate3(nonlinearity(dilate2_out)))
        #dilate3_out = nonlinearity(dilate3_out)
        dilate4_out = self.bn(self.dilate4(nonlinearity(dilate3_out)))
        #dilate4_out = nonlinearity(dilate4_out)
        dilate5_out = self.bn(self.dilate5(nonlinearity(dilate4_out)))
        #dilate5_out = nonlinearity(dilate5_out)
        #out0 = nonlinearity(x - dilate1_out)
        out1 = nonlinearity(dilate2_out - dilate1_out)
        out2 = nonlinearity(dilate3_out - dilate2_out)
        out3 = nonlinearity(dilate4_out - dilate3_out)
        out4 = nonlinearity(dilate5_out - dilate4_out)
        # out5 = nonlinearity(dilate3_out - dilate1_out)
        # out6 = nonlinearity(dilate4_out - dilate1_out)
        # out7 = nonlinearity(dilate5_out - dilate1_out)
        # out8 = nonlinearity(dilate4_out - dilate2_out)
        # out9 = nonlinearity(dilate5_out - dilate2_out)
        # out10 = nonlinearity(dilate5_out - dilate3_out)

        # out = torch.cat([out1, out2, out3, out4], 1)
        # out = nonlinearity(self.bn(self.channelExtension(out)))
        # out = self.channelExtension(out1 + out2 + out3 + out4)
        # out = self.conv(out)
        # out = nonlinearity(self.bn(out))
        out = out1 + out2 + out3 + out4

        # out_gp = nn.AvgPool2d(out.shape[2:])(out)
        # out_gp = self.conv_gp(out_gp)
        # out_gp = self.bn_gp(out_gp)
        #
        # out_att = x_master * out_gp
        # out_att = self.bn(out_att + self.conv(out))

        # out_att = self.ChannelAttention(out, x_master)
        #
        # finalout = nonlinearity(x_gp + out_att)

        finalout = nonlinearity(x_gp + x_master*out)

        return finalout


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

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
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
        #x = self.relu3(x)
        return x


class AttentionPyramidNet(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(AttentionPyramidNet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.dblock = PyramidBlock(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        #self.attention_decoder3 = ChannelAttention(filters[2], filters[1])
        self.attention_decoder3 = SpatialAttention(filters[1],filters[1],filters[0])
        #self.decoder3 = GAU(filters[2], filters[1])
        #self.attention_decoder3 = SKBlock(filters[1])

        self.decoder2 = DecoderBlock(filters[1], filters[0])
        #self.attention_decoder2 = ChannelAttention(filters[1], filters[0])
        self.attention_decoder2 = SpatialAttention(filters[0],filters[0],filters[0]//2)
        #self.decoder2 = GAU(filters[1], filters[0])
        #self.attention_decoder2 = SKBlock(filters[0])

        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes-1, 3, padding=1)

    def forward(self, x):
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

        # Decoder
        # d3 = self.decoder3(e3) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d3 = self.decoder3(e3)
        d3 = self.attention_decoder3(e2, d3)
        #d3 = self.decoder3(e3, e2)

        d2 = self.decoder2(d3)
        d2 = self.attention_decoder2(e1, d2)
        #d2= self.decoder2(d3, e1)

        d1 = self.decoder1(d2)
        d1 = nonlinearity(d1)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
        #return out


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