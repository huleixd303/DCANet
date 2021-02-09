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

class PyramidBlock(nn.Module):
    def __init__(self, channel):
        super(PyramidBlock, self).__init__()


        #global pooling
        #self.conv_gp = nn.Conv2d(channel, channel, 1, 1, 0)
        #self.bn_gp = nn.BatchNorm2d(channel)

        #pyamidblock
        for i in range(1, 5):
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
            #self.add_module(
            #   "respool9_{}".format(i),
            #   nn.Sequential(
            #        nn.Conv2d(channel//4, channel//4, 9, 1, 4),
            #        nn.BatchNorm2d(channel//4),
            #        #nn.MaxPool2d(9, 1, 4)
            #    )
            #)

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

        for i in range(1, 5):
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

            #residual9 = respool9path
            #respool9path = self.__getattr__("respool9_{}".format(i))(respool9path)
            #respool9path = self.relu(residual9 + respool9path)
            #x9 = self.relu(x9)

        sum_conv = respool3path + respool5path + respool7path
        cat_conv = torch.cat([respool3path.unsqueeze(1), respool5path.unsqueeze(1), \
                              respool7path.unsqueeze(1)], 1)
        sum_conv_gp = nn.AvgPool2d(sum_conv.shape[2:])(sum_conv).squeeze_()
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




class PyramidBlockdefeat(nn.Module):
    def __init__(self, channel):
        super(PyramidBlockdefeat,self).__init__()
        self.convcompress = nn.Conv2d(channel, channel//4, 1)
        self.normcompress = nn.BatchNorm2d(channel//4)
        self.relu = nonlinearity



        for i in range(1, 5):
            self.add_module(
                "respool3_{}".format(i),
                nn.MaxPool2d(kernel_size=3 ,stride=1, padding=1)
            )
            self.add_module(
                "resconv3_{}".format(i),
                nn.Sequential(

                    nn.Conv2d(channel//4,channel//4,kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channel//4)))

            self.add_module(
                "respool5_{}".format(i),
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
            )
            self.add_module(
                "resconv5_{}".format(i),
                nn.Sequential(

                    nn.Conv2d(channel//4,channel//4,kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(channel // 4)))

            self.add_module(
                "respool7_{}".format(i),
                nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
            )
            self.add_module(
                "resconv7_{}".format(i),
                nn.Sequential(

                    nn.Conv2d(channel//4,channel//4,kernel_size=7, stride=1, padding=3),
                    nn.BatchNorm2d(channel // 4)))


        self.convre = nn.Conv2d(channel//4, channel, 1)
        self.normre = nn.BatchNorm2d(channel)
        self.relure = nonlinearity


    def forward(self, input):
        x = self.convcompress(input)
        x = self.normcompress(x)
        x = self.relu(x)

        x3,x33,x5,x55,x7,x77= x,x,x,x,x,x
        respool3path,respool5path,respool7path = x3,x5,x7
        resconv3path,resconv5path,resconv7path = x33,x55,x77


        for i in range(1, 5):
            respool3path = self.__getattr__("respool3_{}".format(i))(respool3path)
            x3 = x3 + respool3path
            resconv3path = self.__getattr__("resconv3_{}".format(i))(resconv3path)
            x33 = x33 + resconv3path
            x33 = self.relu(x33)

            respool5path = self.__getattr__("respool5_{}".format(i))(respool5path)
            x5 = x5 + respool5path
            resconv5path = self.__getattr__("resconv5_{}".format(i))(resconv5path)
            x55 =x55 + resconv5path
            x55 = self.relu(x55)

            respool7path = self.__getattr__("respool7_{}".format(i))(respool7path)
            x7 = x7 + respool7path
            resconv7path = self.__getattr__("resconv7_{}".format(i))(resconv7path)
            x77 = x77 + resconv7path
            x77 = self.relu(x77)


        out = x3*x33 + x5*x55 + x7*x77
        out = self.convre(out)
        out = self.normre(out)
        out = self.relure(out)

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


class DlinkPyramidNet(nn.Module):
    def __init__(self, n_classes=2):
        super(DlinkPyramidNet,self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.pyramid = PyramidBlock(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes - 1, 3, padding=1)

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
        e3 = self.pyramid(e3)

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


class DinkNet34_less_pool(nn.Module):
    def __init__(self, n_classes=2):
        #super(DinkNet34_more_dilate, self).__init__()
        super(DinkNet34_less_pool, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.dblock = Dblock_more_dilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
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