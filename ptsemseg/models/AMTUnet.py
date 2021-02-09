import torch.nn as nn
import torch
import torch.nn as nn
from ptsemseg.models.utils import *

class AMTConv(nn.Module):
    def __init__(self, input_dim, output_dim, ):
        super(AMTConv, self).__init__()

        self.branch1x1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)

        self.branch3x3_1 = nn.Conv2d(input_dim, output_dim//2, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(output_dim//2, output_dim, 3, 1, 1)

        self.branch5x5_1 = nn.Conv2d(input_dim, output_dim//2, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(output_dim//2, output_dim, 5, 1, 2)

        self.catconv = nn.Sequential(
            nn.Conv2d(output_dim*3, output_dim, 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 3, 1, 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 3, 1, 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        barnch1x1 = self.branch1x1(x)
        barnch3x3 = self.branch3x3_1(x)
        barnch3x3 = self.branch3x3_2(barnch3x3)
        barnch5x5 = self.branch5x5_1(x)
        barnch5x5 = self.branch5x5_2(barnch5x5)
        branch = self.catconv(torch.cat([barnch1x1, barnch3x3, barnch5x5], dim=1))

        out = self.conv1(branch)
        out = self.conv2(out)

        return out



class AMTUnet(nn.Module):
    def __init__(self, n_classes=2, channel=3):
        super(AMTUnet, self).__init__()
        filters = [32, 64, 128, 256, 512]
        self.in_channels = channel
        self.is_batchnorm = True
        self.is_deconv = True

        self.conv1 = AMTConv(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = AMTConv(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = AMTConv(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes - 1, 1)
        # self.final = nn.Conv2d(filters[0], n_classes, 1) #for Echo

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.output_layer = nn.Sequential(
        #     nn.Conv2d(filters[0], n_classes-1, 1, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)


        final = self.final(up1)
        final = torch.sigmoid(final)

        return final
