import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

from functools import partial

nonlinearity = partial(F.relu, inplace=True)
def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, output_stride, BatchNorm,
                 pretrained=True):
        super(AlignedXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        self.block1 = Block(64, 128, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False,
                            grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, BatchNorm=BatchNorm,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(2048)

        # Init weights
        self._init_weight()

        # Load pretrained model
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in model_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)



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
        #self.channelExtension = nn.Conv2d(channel*4, channel, kernel_size=1, dilation=1, padding=0)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16, bias=False)
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
        dilate2_out = self.bn(self.dilate2((dilate1_out)))
        #dilate2_out = nonlinearity(dilate2_out)
        dilate3_out = self.bn(self.dilate3((dilate2_out)))
        #dilate3_out = nonlinearity(dilate3_out)
        dilate4_out = self.bn(self.dilate4((dilate3_out)))
        #dilate4_out = nonlinearity(dilate4_out)
        dilate5_out = self.bn(self.dilate5((dilate4_out)))
        #dilate5_out = nonlinearity(dilate5_out)
        #out0 = nonlinearity(x - dilate1_out)
        out1 = nonlinearity(dilate2_out - dilate1_out)
        out2 = nonlinearity(dilate3_out - dilate2_out)
        out3 = nonlinearity(dilate4_out - dilate3_out)
        out4 = nonlinearity(dilate5_out - dilate4_out)

        out = out1 + out2 + out3 + out4



        finalout = nonlinearity(x_gp + x_master*out)

        return finalout

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        if in_channels == 2048:
            self.conv1 = nn.Conv2d(in_channels, in_channels // 16, 1, bias=False)
            self.norm1 = nn.BatchNorm2d(in_channels // 16)
            self.relu1 = nonlinearity

            self.deconv2 = nn.ConvTranspose2d(in_channels // 16, in_channels // 16, 3, stride=2, padding=1,
                                              output_padding=1)
            self.norm2 = nn.BatchNorm2d(in_channels // 16)
            self.relu2 = nonlinearity

            self.conv3 = nn.Conv2d(in_channels // 16, n_filters, 1, bias=False)
            self.norm3 = nn.BatchNorm2d(n_filters)
            self.relu3 = nonlinearity

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1,bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1, bias=False)
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


class Difxception(nn.Module):
    def __init__(self, n_classes=2):
        super(Difxception, self).__init__()

        filters = [64, 128, 256, 2048]

        xception = AlignedXception(output_stride = 16, \
                                   BatchNorm = nn.BatchNorm2d, \
                                   pretrained=True)

        self.firstconv = xception.conv1
        self.bn1 = xception.bn1
        self.relu = xception.relu
        self.conv2 = xception.conv2
        self.bn2 = xception.bn2
        self.block1 = xception.block1
        self.block2 = xception.block2
        self.block3 = xception.block3


        self.block4 = xception.block4
        self.block5 = xception.block5
        self.block6 = xception.block6
        self.block7 = xception.block7
        self.block8 = xception.block8
        self.block9 = xception.block9
        self.block10 = xception.block10
        self.block11 = xception.block11
        self.block12 = xception.block12
        self.block13 = xception.block13
        self.block14 = xception.block14
        self.block15 = xception.block15
        self.block16 = xception.block16
        self.block17 = xception.block17
        self.block18 = xception.block18
        self.block19 = xception.block19


        self.block20 = xception.block20
        self.conv3 = xception.conv3
        self.bn3 = xception.bn3
        self.conv4 = xception.conv4
        self.bn4 = xception.bn4
        self.conv5 = xception.conv5
        self.bn5 = xception.bn5


        self.dblock = DifPyramidBlock(2048)


        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes-1, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        block1 = self.block1(x)
        block1 = self.relu(block1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)

        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7)
        block9 = self.block9(block8)
        block10 = self.block10(block9)
        block11 = self.block11(block10)
        block12 = self.block12(block11)
        block13 = self.block13(block12)
        block14 = self.block14(block13)
        block15 = self.block15(block14)
        block16 = self.block16(block15)
        block17 = self.block17(block16)
        block18 = self.block18(block17)
        block19 = self.block19(block18)


        block20 = self.block20(block19)
        block20 = self.relu(block20)

        block20 = self.conv3(block20)
        block20 = self.bn3(block20)
        block20 = self.relu(block20)

        block20 = self.conv4(block20)
        block20 = self.bn4(block20)
        block20 = self.relu(block20)

        block20 = self.conv5(block20)
        block20 = self.bn5(block20)
        block20 = self.relu(block20)

        # Center
        e3 = self.dblock(block20)

        # Decoder
        d3 = self.decoder3(e3) + block2
        d2 = self.decoder2(d3) + block1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)