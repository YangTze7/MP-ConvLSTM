## MP_CLSTM
# -*- coding: utf-8 -*-
# @Author: Chang-jiang.Shi
# @Date:   2021-08-17 20:14:18
# @Last Modified by:   Chang-jiang.Shi
# @Last Modified time: 2022-10-16 21:41:51

import numpy as np
import torch
from torch import nn
from torch.nn.modules import padding
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.parameter import Parameter

from convlstm import ConvLSTM, ConvLSTMCell
from non_local_concatenation import NONLocalBlock2D


class Channel_Attention(nn.Module):
    def __init__(self, channel, r):
        super(Channel_Attention, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel // r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel // r, channel, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)

        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1 + y2)
        return x * y


class Channel_Attention(nn.Module):
    def __init__(self, channel, r):
        super(Channel_Attention, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel // r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel // r, channel, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)

        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1 + y2)
        return x * y


class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size):
        super(Spatial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return x * mask


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,
                              1,
                              kernel_size=k_size,
                              padding=(k_size - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1,
                                              -2)).transpose(-1,
                                                             -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class MP_CLSTM(nn.Module):
    def __init__(self, in_channel, r, w):
        super(MP_CLSTM, self).__init__()

        new_channel = in_channel // r

        self.ca3 = eca_layer()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel * 1 * 1,
                      new_channel,
                      kernel_size=3,
                      padding=3 // 2), nn.BatchNorm2d(new_channel), nn.ReLU())
        self.convlstm1 = ConvLSTM(new_channel * 1, new_channel * 1, (3, 3), 1,
                                  True, True, False)

        self.bn2 = nn.BatchNorm3d(new_channel * 1)

        self.convlstm2 = ConvLSTM(new_channel * 1, new_channel * 1, (3, 3), 1,
                                  True, True, False)

        self.bn1 = nn.BatchNorm3d(new_channel * 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(new_channel * 1 * 1,
                      new_channel * 1,
                      kernel_size=3,
                      padding=3 // 2), nn.BatchNorm2d(new_channel * 1),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(new_channel * 1 * 1,
                      new_channel * 1,
                      kernel_size=3,
                      padding=3 // 2), nn.BatchNorm2d(new_channel * 1),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(new_channel * 1 * 1 * 4,
                      new_channel * 1 * 1,
                      kernel_size=3,
                      stride=1,
                      padding=(1, 1)), nn.BatchNorm2d(new_channel * 1 * 1),
            nn.ReLU())
        self.conv31 = nn.Sequential(
            nn.Conv2d(new_channel * 1 * 1,
                      new_channel * 1,
                      kernel_size=3,
                      stride=1,
                      padding=(1, 1)), nn.BatchNorm2d(new_channel * 1 * 1),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(new_channel * 1,
                      2,
                      kernel_size=3,
                      stride=1,
                      padding=(1, 1)), nn.BatchNorm2d(2), nn.ReLU())

        self.classifier = nn.Sequential(nn.Dropout(p=0.5), self.conv31,
                                        nn.Dropout(p=0.5), self.conv4,
                                        nn.ReLU(),
                                        nn.AvgPool2d(kernel_size=w, stride=1))

    def forward_one(self, t1):

        x = self.conv0(t1)
        x = torch.unsqueeze(x, 1)

        return (x)

    def forward(self, t1, t2):

        t1 = self.forward_one(t1)
        t2 = self.forward_one(t2)

        x = torch.cat((t1, t2), dim=1)

        x1, last_states1 = self.convlstm1(x)
        x1 = self.bn1(torch.transpose(x1[0], 1, 2))
        x1 = torch.transpose(x1, 1, 2)

        x2, last_states2 = self.convlstm2(x1)
        x2 = self.bn2(torch.transpose(x2[0], 1, 2))
        x2 = torch.transpose(x2, 1, 2)

        x21 = x2[:, 0, :, :, :]  #h0
        x22 = x2[:, 1, :, :, :]  #h1

        x31 = self.conv1(x21)  #h0'
        x32 = self.conv2(x22)  #h1'

        x4 = torch.cat((x1[:, 0, :, :, :], x1[:, 1, :, :, :], x31, x32), dim=1)
        x5 = self.ca3(x4)

        x6 = self.conv3(x5)

        x7 = self.classifier(x6)

        logits = x7

        return logits


if __name__ == '__main__':

    t1 = torch.rand((8, 1, 12, 64, 64))
    t2 = torch.rand((8, 1, 12, 64, 64))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    t1 = torch.rand((8, 12, 5, 5)).to(device)
    t2 = torch.rand((8, 12, 5, 5)).to(device)

    gt = torch.ones(8, 5, 5)

    model = MP_CLSTM(12, 6, 5).to(device)

    logits = model(t1, t2)
    y_pred = logits.argmax(1)
    print(y_pred)

    print("end")
