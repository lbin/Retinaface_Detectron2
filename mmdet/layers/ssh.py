import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import Conv2d, get_norm

from aifwdet.layers.DCNv2.dcn_v2 import DCN

# from detectron2.layers.batch_norm import NaiveSyncBatchNorm


def conv_bn(in_channel, out_channel, stride=1, leaky=0, norm="BN"):
    return nn.Sequential(
        Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channel),
        ),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(in_channel, out_channel, stride, norm="BN"):
    return Conv2d(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        norm=get_norm(norm, out_channel),
    )


def conv_bn1X1(in_channel, out_channel, stride, leaky=0, norm="BN"):
    return nn.Sequential(
        Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
            norm=get_norm(norm, out_channel),
        ),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_dw(in_channel, out_channel, stride, leaky=0.1, norm="BN"):
    return nn.Sequential(
        Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=in_channel,
            norm=get_norm(norm, out_channel),
        ),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        Conv2d(
            in_channel,
            out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=get_norm(norm, out_channel),
        ),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class DeformConv(nn.Module):
    def __init__(self, in_channel, out_channel, norm="BN"):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(get_norm(norm, out_channel), nn.ReLU(inplace=True))
        self.conv = DCN(
            in_channel,
            out_channel,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


# SSH:Single Stage Headless Face Detector
class SSH(nn.Module):
    def __init__(self, cfg, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        self.use_dcnv2 = cfg.MODEL.RETINANET.WITH_DCNv2
        self.norm = cfg.MODEL.RETINANET.NORM
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv_1 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1, norm=self.norm)

        self.conv_2 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky, norm=self.norm)
        self.conv_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1, norm=self.norm)

        self.conv_4 = conv_bn(
            out_channel // 4, out_channel // 4, stride=1, leaky=leaky, norm=self.norm
        )
        self.conv_5 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1, norm=self.norm)
        if self.use_dcnv2:
            self.dcn = DeformConv(in_channel, out_channel, norm=self.norm)

    def forward(self, input):
        conv_1 = self.conv_1(input)

        conv_2 = self.conv_2(input)
        conv_2_3 = self.conv_3(conv_2)

        conv_4 = self.conv_4(conv_2)
        conv_4_5 = self.conv_5(conv_4)

        out = torch.cat([conv_1, conv_2_3, conv_4_5], dim=1)
        out = F.relu(out)
        if self.use_dcnv2:
            out = self.dcn(out)
        return out
