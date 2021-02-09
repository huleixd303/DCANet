import torch.nn.functional as F
import torch
from ptsemseg.models.utils import *

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=5):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv2d = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.BNRELU = nn.Sequential(
            # nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.conv_1x1 = nn.Conv2d(self.ch_in, self.ch_out, 1, bias=False)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(ch_out),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        x = self.conv_1x1(x)
        for i in range(self.t):

            if i == 0:
                x = self.conv2d(x)
                x1 = self.BNRELU(x)

            elif i == 4:
                x1 = self.conv2d(x1) + x

            x1 = self.conv2d(x1) + x
            x1 = self.BNRELU(x1)
        return x1

class RCNNUNet(nn.Module):
    def __init__(self, n_classes=2, is_deconv=True,):
        super(RCNNUNet, self).__init__()

        self.nclasses = n_classes
        self.is_deconv = is_deconv
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.RRCNN1 = Recurrent_block(ch_in=3, ch_out=64, t=5)

        self.RRCNN2 = Recurrent_block(ch_in=64, ch_out=128, t=5)

        self.RRCNN3 = Recurrent_block(ch_in=128, ch_out=256, t=5)

        self.center = Recurrent_block(ch_in=256, ch_out=512, t=5)

        self.up_concat3 = unetUp(512, 256, self.is_deconv)
        self.cat_3 = Recurrent_block(ch_in=256, ch_out=256, t=5)

        self.up_concat2 = unetUp(256, 128, self.is_deconv)
        self.cat_2 = Recurrent_block(ch_in=128, ch_out=128, t=5)

        self.up_concat1 = unetUp(128, 64, self.is_deconv)
        self.cat_1 = Recurrent_block(ch_in=64, ch_out=64, t=5)

        self.final = nn.Conv2d(64, n_classes - 1, 1)

    def forward(self, input):
        conv1 = self.RRCNN1(input)
        del input
        maxpool1 = self.Maxpool(conv1)

        conv2 = self.RRCNN2(maxpool1)
        del maxpool1
        maxpool2 = self.Maxpool(conv2)

        conv3 = self.RRCNN3(maxpool2)
        del maxpool2
        maxpool3 = self.Maxpool(conv3)

        center = self.center(maxpool3)
        del maxpool3

        up3 = self.up_concat3(conv3, center)
        del center
        del conv3
        up3 = self.cat_3(up3)

        up2 = self.up_concat2(conv2, up3)
        del conv2
        up2 = self.cat_2(up2)

        up1 = self.up_concat1(conv1, up2)
        del conv1
        up1 = self.cat_1(up1)

        final = self.final(up1)
        final = torch.sigmoid(final)

        return final
