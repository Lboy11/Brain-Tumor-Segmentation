"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.ensemble.modules import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, width_multiplier=1, trilinear=True, conv_type=conv_para, num_modalities=4,
        parallel=False, exchange=False, feature=False):
        super(UNet, self).__init__()
        _channels = (32, 64, 128, 256, 512)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c * width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.conv_type = conv_type
        self.parallel = parallel
        self.feature = feature

        self.inc = DoubleConv(
            n_channels, self.channels[0], conv_type=self.conv_type, num_modalities=num_modalities, exchange=exchange)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.conv_type,
                          num_modalities=num_modalities, exchange=exchange)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.conv_type,
                          num_modalities=num_modalities, exchange=exchange)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.conv_type,
                          num_modalities=num_modalities, exchange=exchange)
        factor = 2 if trilinear else 1
        self.down4 = Down(self.channels[3], self.channels[4] // factor,
                          conv_type=self.conv_type, num_modalities=num_modalities, exchange=exchange)
        self.up1 = Up(self.channels[4], self.channels[3] // factor, trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.up2 = Up(self.channels[3], self.channels[2] // factor, trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.up3 = Up(self.channels[2], self.channels[1] // factor, trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.up4 = Up(self.channels[1], self.channels[0], trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.outc = OutConv(
            self.channels[0], n_classes, conv_type=self.conv_type)

    def forward(self, x: list, modality=0):
        x1 = self.inc(x, modality=modality)
        x2 = self.down1(x1, modality=modality)

        x3 = self.down2(x2, modality=modality)

        x4 = self.down3(x3, modality=modality)

        x5 = self.down4(x4, modality=modality)
        c5d = x5


        x = self.up1(x5, x4, modality=modality)

        x = self.up2(x, x3, modality=modality)

        x = self.up3(x, x2, modality=modality)

        x = self.up4(x, x1, modality=modality)

        logits = self.outc(x)


        return logits, c5d



