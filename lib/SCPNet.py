import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import argparse
import numpy as np
from torch import nn
from torch.nn import init
from os.path import join
from lib.EfficientNet import EfficientNet

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


                                          

class TPM(nn.Module):
    def __init__(self, in_dim):
        super(TPM, self).__init__()

        # Posting-H
        self.query_conv_h = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1)) 
        self.key_conv_h = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
        self.value_conv_h = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))

        # Posting-L
        self.query_conv_L = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
        self.key_conv_L = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
        self.value_conv_L = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))

        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        # finally refine
        self.conv_final = nn.Conv2d(2 * 64, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B1, C1, H1, W1 = x.size()
        xh_q = self.query_conv_h(x).view(B1, -1, W1 * H1)
        xh_k = self.key_conv_h(x).view(B1, -1, W1 * H1)
        xh_v = self.value_conv_h(x).view(B1, -1, W1 * H1)

        xl_q = self.query_conv_L(x).view(B1, -1, W1 * H1)
        xl_k = self.key_conv_L(x).view(B1, -1, W1 * H1)
        xl_v = self.value_conv_L(x).view(B1, -1, W1 * H1)

        xh_hw = torch.bmm(xh_q.permute(0, 2, 1), xh_k)  # HW x HW
        xh_hws = self.softmax(xh_hw)
        xh1 = torch.bmm(xh_v, xh_hws)  # C X H X W
        xh1 = xh1.view(B1, 64, H1, W1)
        xh1 = self.g1 * xh1

        xl_hw = torch.bmm(xl_q.permute(0, 2, 1), xl_k)  # HW x HW
        xl_hws = self.softmax(xl_hw)
        xl1 = torch.bmm(xl_v, xl_hws)  # C X H X W
        xl1 = xl1.view(B1, 64, H1, W1)
        xl1 = self.g2 * xl1

        out_final = self.conv_final(torch.cat((xh1, xl1), dim=1))
        return out_final


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()

        self.conv_1 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv_3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv_4 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = BasicConv2d(64, 1, kernel_size=1)

        self.sa = SpatialAttention()

    def forward(self, x1, x2, fg):
        if x2.size()[2:] != x1.size()[2:]:
            x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear')
        if fg.size()[2:] != x1.size()[2:]:
            fg = F.interpolate(fg, size=x1.size()[2:], mode='bilinear')
        x1 = self.conv_1(x1)
        x2 = self.conv_2(x2)
        x1_1 = self.conv_3(x1 + x2)
        xc = self.conv_4(x1_1 * fg)
        out = self.sa(xc) + xc
        return out


class ASPP(nn.Module):
    def __init__(self, outchannel):
        super(ASPP, self).__init__()
        self.conv1 = BasicConv2d(outchannel, outchannel, kernel_size=3, padding=1, dilation=1)
        self.conv2 = BasicConv2d(outchannel, outchannel, kernel_size=3, padding=3, dilation=3)
        self.conv3 = BasicConv2d(outchannel, outchannel, kernel_size=3, padding=5, dilation=5)
        self.conv0 = BasicConv2d(outchannel, outchannel, kernel_size=1)
        self.conv = BasicConv2d(4 * outchannel, outchannel, kernel_size=1)


    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = torch.cat((x0, x1, x2, x3), dim=1)
        out = self.conv(out)
        out = out + x
        return out



class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 4, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // 4, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)



class CPM(nn.Module):
    def __init__(self, channel):
        super(CPM, self).__init__()
        self.conv_1 = BasicConv2d(channel, 64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.cbr = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True)
                                 )

        self.aspp = ASPP(64)
        self.conv192_64 = nn.Conv2d(192, 64, 1, stride=1)
        self.ca = ChannelAttention(64)

    def forward(self, x, xe, xg):

        if xg.size()[2:] != x.size()[2:]:
            xg = F.interpolate(xg, size=x.size()[2:], mode='bilinear')
        if xe.size()[2:] != x.size()[2:]:
            xe = F.interpolate(xe, size=x.size()[2:], mode='bilinear')

        x1 = xg
        x2 = xe
        xc = self.conv192_64(torch.cat((x1, x2, x), dim=1))
        out1 = self.aspp(xc)
        out = self.ca(out1) * out1
        final = out + out1
        return final

class Network(nn.Module):
    # EfficientNet based encoder decoder
    def __init__(self, channel=64, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.context_encoder = EfficientNet.from_pretrained('efficientnet-b1')

        self.reduce4 = BasicConv2d(320, 64, kernel_size=1)
        self.reduce3 = BasicConv2d(112, 64, kernel_size=1)
        self.reduce2 = BasicConv2d(40, 64, kernel_size=1)
        self.reduce1 = BasicConv2d(24, 64, kernel_size=1)
        self.cpm4 = CPM(64)
        self.cpm3 = CPM(64)
        self.cpm2 = CPM(64)
        self.cpm1 = CPM(64)

        self.e = SAM()

        self.g = TPM(64)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv64_1 = nn.Conv2d(64, 1, 1)

        self.decoder_conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
                                           nn.ReLU(inplace=True))
        self.decoder_conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
                                           nn.ReLU(inplace=True))
        self.decoder_conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
                                           nn.ReLU(inplace=True))

    def forward(self, x):
        # backbone
        endpoints = self.context_encoder.extract_endpoints(x)
        r1 = endpoints['reduction_2']
        r2 = endpoints['reduction_3']
        r3 = endpoints['reduction_4']
        r4 = endpoints['reduction_5']

        r1 = self.reduce1(r1)
        r2 = self.reduce2(r2)
        r3 = self.reduce3(r3)
        r4 = self.reduce4(r4)

        # ----CRM---#
        f_g = self.g(r4)
        # ---BEM---#
        f_e = self.e(r1, r2, f_g)

        # Decoder
        f4 = self.cpm4(r4, f_e, f_g)
        f3 = self.cpm3(r3, f_e, f_g)
        f2 = self.cpm2(r2, f_e, f_g)
        f1 = self.cpm1(r1, f_e, f_g)


        S_3 = self.decoder_conv1(f3 + self.upsample(f4))
        S_2 = self.decoder_conv2(f2 + self.upsample(S_3))
        S_1 = self.decoder_conv3(f1 + self.upsample(S_2))

        S_4_pred = F.interpolate(self.conv64_1(f4), scale_factor=32,
                                 mode='bilinear')
        S_3_pred = F.interpolate(self.conv64_1(S_3), scale_factor=16,
                                 mode='bilinear')
        S_2_pred = F.interpolate(self.conv64_1(S_2), scale_factor=8,
                                 mode='bilinear')
        S_1_pred = F.interpolate(self.conv64_1(S_1), scale_factor=4,
                                 mode='bilinear')
        return S_4_pred, S_3_pred, S_2_pred, S_1_pred


		
