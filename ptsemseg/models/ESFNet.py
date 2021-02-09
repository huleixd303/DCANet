import torch
import torch.nn as nn
import torch.nn.functional as F

class ESFNet(nn.Module):
    def __init__(self,
                 n_classes = 2,
                 down_factor=8,
                 interpolate=True,
                 dilation=True,
                 dropout=False,
                 ):
        super(ESFNet, self).__init__()

        self.name = 'ESFNet_base'
        self.nb_classes = n_classes
        self.down_factor = down_factor
        self.interpolate = interpolate
        self.stage_channels = [-1, 16, 64, 128, 256, 512]

        if dilation == True:
            self.dilation_list = [1, 2, 4, 8, 16]
        else:
            self.dilation_list = [1, 1, 1, 1, 1 ]

        if dropout == True:
            self.dropout_list = [0.01, 0.001]

        if down_factor==8:
            # 8x downsampling
            self.encoder = nn.Sequential(
                down_sampling_block(3, 16),

                DownSamplingBlock_v2(in_channels=self.stage_channels[1], out_channels=self.stage_channels[2]),

                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0], dropout_rate=0.0),

                DownSamplingBlock_v2(in_channels=self.stage_channels[2], out_channels=self.stage_channels[3]),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[1], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[2], dropout_rate=0.0),

                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[3], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[4], dropout_rate=0.0),

            )
            if interpolate == True:
                self.project_layer = nn.Conv2d(self.stage_channels[3], self.nb_classes, 1, bias=False)
            else:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(self.stage_channels[3], self.stage_channels[2], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[2]),
                    nn.ReLU(inplace=True),
                    SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                    SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                    nn.ConvTranspose2d(self.stage_channels[2], self.stage_channels[1], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[1]),
                    nn.ReLU(inplace=True),
                    SFRB(in_channels=self.stage_channels[1], out_channels=self.stage_channels[1], ),
                    SFRB(in_channels=self.stage_channels[1], out_channels=self.stage_channels[1], ),
                    nn.ConvTranspose2d(self.stage_channels[1], self.nb_classes, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=False)

                )

        elif down_factor==16:
            # 16x downsampling
            self.encoder = nn.Sequential(
                down_sampling_block(3, 16),
                DownSamplingBlock_v2(self.stage_channels[1], self.stage_channels[2]),

                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0], dropout_rate=0.0),

                DownSamplingBlock_v2(self.stage_channels[2], self.stage_channels[3]),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], dilation=self.dilation_list[0], dropout_rate=0.0),

                DownSamplingBlock_v2(self.stage_channels[3], self.stage_channels[4]),
                SFRB(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], dilation=self.dilation_list[1], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], dilation=self.dilation_list[2], dropout_rate=0.0),

                SFRB(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], dilation=self.dilation_list[3], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], dilation=self.dilation_list[0], dropout_rate=0.0),
                SFRB(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], dilation=self.dilation_list[4], dropout_rate=0.0),
            )
            if interpolate == True:
                self.project_layer = nn.Conv2d(self.stage_channels[4], self.nb_classes, 1, bias=False)
            else:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(self.stage_channels[4], self.stage_channels[3], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[3]),
                    nn.ReLU(inplace=True),
                    SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3]),
                    SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3]),
                    nn.ConvTranspose2d(self.stage_channels[3], self.stage_channels[2], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[2]),
                    nn.ReLU(inplace=True),
                    SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2]),
                    SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2]),
                    nn.ConvTranspose2d(self.stage_channels[2], self.stage_channels[1], kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[1]),
                    nn.ReLU(inplace=True),
                    SFRB(in_channels=self.stage_channels[1], out_channels=self.stage_channels[1]),
                    SFRB(in_channels=self.stage_channels[1], out_channels=self.stage_channels[1]),
                    nn.ConvTranspose2d(self.stage_channels[2], self.nb_classes, kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                )


    def forward(self, input):

        encoder_out = self.encoder(input)

        if self.interpolate == True:
            decoder_out = self.project_layer(encoder_out)
            decoder_out = F.interpolate(decoder_out, scale_factor=self.down_factor, mode='bilinear', align_corners=True)
        else:
            decoder_out = self.decoder(encoder_out)

        # decoder_out = F.sigmoid(decoder_out) #for dice+bce loss
        return decoder_out


class ESFNet_mini_ex(nn.Module):
    def __init__(self,
                 config,
                 interpolate=True,
                 dilation=True,
                 dropout=False,):
        super(ESFNet_mini_ex, self).__init__()

        self.name = 'ESFNet_mini_ex'
        self.nb_classes = config.nb_classes
        self.interpolate = interpolate
        self.stage_channels = [-1, 16, 64, 128, 256, 512]

        if dilation == True:
            self.dilation_list = [1, 2, 4, 8, 16]
        else:
            self.dilation_list = [1, 1, 1, 1, 1]


        self.encoder = nn.Sequential(
            down_sampling_block(3, 16),
            DownSamplingBlock_v2(in_channels=self.stage_channels[1], out_channels=self.stage_channels[2]),

            SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2],
                 dilation=self.dilation_list[0], dropout_rate=0.0),
            SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2],
                 dilation=self.dilation_list[0], dropout_rate=0.0),
            SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2],
                 dilation=self.dilation_list[0], dropout_rate=0.0),
            DownSamplingBlock_v2(in_channels=self.stage_channels[2], out_channels=self.stage_channels[3]),
            SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3],
                 dilation=self.dilation_list[1], dropout_rate=0.0),
            SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3],
                 dilation=self.dilation_list[2], dropout_rate=0.0),
            SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3],
                 dilation=self.dilation_list[3], dropout_rate=0.0),
            SFRB(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3],
                 dilation=self.dilation_list[4], dropout_rate=0.0),
        )
        if interpolate == True:
            self.project_layer = nn.Conv2d(self.stage_channels[3], self.nb_classes, 1, bias=False)
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.stage_channels[3], self.stage_channels[2], kernel_size=3, stride=2,
                                   padding=1,
                                   output_padding=1, bias=False),
                nn.BatchNorm2d(self.stage_channels[2]),
                nn.ReLU(inplace=True),
                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                SFRB(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                nn.ConvTranspose2d(self.stage_channels[2], self.stage_channels[1], kernel_size=3, stride=2,
                                   padding=1,
                                   output_padding=1, bias=False),
                nn.BatchNorm2d(self.stage_channels[1]),
                nn.ReLU(inplace=True),
                SFRB(in_channels=self.stage_channels[1], out_channels=self.stage_channels[1], ),
                SFRB(in_channels=self.stage_channels[1], out_channels=self.stage_channels[1], ),
                nn.ConvTranspose2d(self.stage_channels[1], self.nb_classes, kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=False)
            )
    def forward(self, x):

        encoder_out = self.encoder(input)

        if self.interpolate == True:
            decoder_out = self.project_layer(encoder_out)
            decoder_out = F.interpolate(decoder_out, scale_factor=8, mode='bilinear', align_corners=True)
        else:
            decoder_out = self.decoder(encoder_out)

        return decoder_out

class Separabel_conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 kernel_size=(3,3),
                 dilation=(1,1),
                 #padding=(1,1),
                 stride=(1,1),
                 bias=False):
        """
        # Note: Default for kernel_size=3,
        for Depthwise conv2d groups should equal to in_channels and out_channels == in_channels
        Only bn after depthwise_conv2d and no no-linear

        padding = (kernel_size-1) / 2
        padding = padding * dilation
        """
        super(Separabel_conv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding= (int((kernel_size[0]-1)/2)*dilation[0],int((kernel_size[1]-1)/2)*dilation[1]),
            dilation= dilation, groups=groups,bias=bias
        )
        self.dw_bn = nn.BatchNorm2d(out_channels)
        self.pointwise_conv2d = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1, padding=0, dilation=1, groups=1, bias=False
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)
    def forward(self, input):

        out = self.depthwise_conv2d(input)
        out = self.dw_bn(out)
        out = self.pointwise_conv2d(out)
        out = self.pw_bn(out)

        return out

class down_sampling_block(nn.Module):

    def __init__(self, inpc, oupc):
        super(down_sampling_block, self).__init__()
        self.branch_conv = nn.Conv2d(inpc, oupc-inpc, 3, stride=2, padding= 1, bias=False)
        self.branch_mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(oupc, eps=1e-03)

    def forward(self, x):
        output = torch.cat([self.branch_conv(x), self.branch_mp(x)], 1)
        output = self.bn(output)

        return F.relu(output)

def channel_shuffle(input, groups):
    """
    # Note that groups set to channels by default
    if depthwise_conv2d means groups == in_channels thus, channels_shuffle doesn't work for it.
    """
    batch_size, channels, height, width = input.shape
    #groups = channels
    channels_per_group = channels // groups

    input = input.view(batch_size, groups, channels_per_group, height, width)
    input = input.transpose(1,2).contiguous()
    input = input.view(batch_size, -1, height, width)

    return input


class DownSamplingBlock_v2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,):
        """
        # Note: Initial_block from ENet
        Default that out_channels = 2 * in_channels
        compared to downsamplingblock_v1, change conv3x3 into Depthwise and projection_layer

        Add: channel_shuffle after concatenate
        to be testing

        gc prelu/ relu
        """
        super(DownSamplingBlock_v2, self).__init__()

        # MaxPooling or AvgPooling
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        '''
        # FacDW
        self.depthwise_conv2d_1 = nn.Conv2d(in_channels= in_channels, out_channels=in_channels, kernel_size=(3,1), stride=(2,1),
                                            padding=(1,0), groups=in_channels, bias=False)
        self.dwbn1 = nn.BatchNorm2d(in_channels)
        self.depthwise_conv2d_2 = nn.Conv2d(in_channels= in_channels, out_channels=in_channels, kernel_size=(1,3), stride=(1,2),
                                            padding=(0,1), groups=in_channels, bias=False)
        self.dwbn2 = nn.BatchNorm2d(in_channels)
        '''
        self.depthwise_conv2d = nn.Conv2d(in_channels= in_channels, out_channels=in_channels,
                                          kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels)

        self.project_layer = nn.Conv2d(in_channels= in_channels, out_channels=out_channels-in_channels,
                                       kernel_size=1, bias=False)
        # here dont need project_bn, need to bn with ext_branch
        #self.project_bn = nn.BatchNorm2d(out_channels-in_channels)

        self.ret_bn = nn.BatchNorm2d(out_channels)
        self.ret_prelu = nn.ReLU(inplace= True)

    def forward(self, input):

        ext_branch = self.pooling(input)
        '''
        # facDW
        main_branch = self.dwbn1(self.depthwise_conv2d_1(input))
        main_branch = self.dwbn2(self.depthwise_conv2d_2(main_branch))

        '''
        main_branch = self.depthwise_conv2d(input)
        main_branch = self.dw_bn(main_branch)

        main_branch = self.project_layer(main_branch)

        ret = torch.cat([ext_branch, main_branch], dim=1)
        ret = self.ret_bn(ret)

        #ret = channel_shuffle(ret, 2)

        return self.ret_prelu(ret)



class SFRB(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 dropout_rate =0.0,
                 ):

        # default decoupled
        super(SFRB, self).__init__()

        self.internal_channels = in_channels // 4
        # compress conv
        self.conv1 = nn.Conv2d(in_channels, self.internal_channels, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        # Depthwise_conv 3x1 and 1x3
        self.conv2 = nn.Conv2d(self.internal_channels, self.internal_channels, (kernel_size,1), stride=(stride,1),
                               padding=(int((kernel_size-1)/2*dilation),0), dilation=(dilation,1),
                               groups=self.internal_channels, bias=False)
        self.conv2_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv3 = nn.Conv2d(self.internal_channels, self.internal_channels, (1,kernel_size), stride=(1,stride),
                               padding=(0,int((kernel_size-1)/2*dilation)), dilation=(1, dilation),
                               groups=self.internal_channels, bias=False)
        self.conv3_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv4 = nn.Conv2d(self.internal_channels, out_channels, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(out_channels)

        # regularization
        self.dropout = nn.Dropout2d(inplace=True, p=dropout_rate)
    def forward(self, input):

        residual = input
        main = self.conv1(input)
        main = self.conv1_bn(main)
        main = F.relu(main, inplace=True)

        main = self.conv2(main)
        main = self.conv2_bn(main)
        main = self.conv3(main)
        main = self.conv3_bn(main)
        main = self.conv4(main)
        main = self.conv4_bn(main)

        if self.dropout.p != 0:
            main = self.dropout(main)

        return F.relu(torch.add(main, residual), inplace=True)