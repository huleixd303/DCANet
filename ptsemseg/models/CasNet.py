import torch.nn.functional as F

import torch
from ptsemseg.models.utils import *
# nonlinearity = partial(F.relu, inplace=True)

class DecoderBlock3(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock3, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.relu2 =nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, n_filters, 3, 1, padding=2, dilation=2, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

        self.unpool = nn.MaxUnpool2d(2, 2)

        # self.conv4 = nn.Conv2d(n_filters*2, n_filters, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs, indices, output_shape):
        x = self.unpool(input=inputs, indices=indices, output_size=output_shape)

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


class DecoderBlock2(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock2, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels, n_filters, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(n_filters)
        self.relu2 =nn.ReLU(inplace=True)

        self.unpool = nn.MaxUnpool2d(2, 2)

        # self.conv3 = nn.Conv2d(in_channels, n_filters, 3, 1, 1, bias=False)
        # self.norm3 = nn.BatchNorm2d(n_filters)
        # self.relu3 = nn.ReLU(inplace=True)

        # self.conv4 = nn.Conv2d(n_filters*2, n_filters, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs, indices, output_shape):
        x = self.unpool(input=inputs, indices=indices, output_size=output_shape)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        # x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)

        # x = self.conv3(x)
        # x = self.norm3(x)
        # x = self.relu3(x)

        return x

class CasNet(nn.Module):
    def __init__(
        self,

        n_classes=2,
        is_unpooling=True,
        in_channels=3

    ):
        super(CasNet, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown2(self.in_channels, 32)
        self.down2 = segnetDown2(32, 64)
        self.down3 = segnetDown3(64, 128)
        self.down4 = segnetDown3(128, 256)
        # self.down5 = segnetDown3(512, 512)

        # self.up5 = segnetUp3(512, 512)
        self.up4 = DecoderBlock3(256, 128)
        self.up3 = DecoderBlock3(128, 64)
        self.up2 = DecoderBlock2(64, 32)
        self.up1 = DecoderBlock2(32, 32)

        self.finaconv = (nn.Conv2d(32, n_classes-1, 1, bias=False))  ## target extraction

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        # down5, indices_5, unpool_shape5 = self.down5(down4)

        # up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(down4, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        up1 = self.finaconv(up1)
        up1 = F.sigmoid(up1)

        return up1














