import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class BnReluConv(nn.Module):
    """docstring for BnReluConv"""
    def __init__(self, inChannels, outChannels, kernelSize=1, stride=1, padding=0):
        super(BnReluConv, self).__init__()
        self.bn = nn.BatchNorm2d(inChannels)
        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    """docstring for ConvBlock"""
    def __init__(self, inChannels, outChannels):
        super(ConvBlock, self).__init__()
        self.cbr1 = BnReluConv(inChannels, outChannels//2, 1, 1, 0)
        self.cbr2 = BnReluConv(outChannels//2, outChannels//2, 3, 1, 1)
        self.cbr3 = BnReluConv(outChannels//2, outChannels, 1, 1, 0)

    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        return x


class SkipLayer(nn.Module):
    """docstring for SkipLayer"""
    def __init__(self, inChannels, outChannels):
        super(SkipLayer, self).__init__()
        if inChannels == outChannels:
            self.conv = None
        else:
            self.conv = nn.Conv2d(inChannels, outChannels, 1)

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x


class Residual(nn.Module):
    """docstring for Residual"""
    def __init__(self, inChannels, outChannels):
        super(Residual, self).__init__()
        self.cb = ConvBlock(inChannels, outChannels)
        self.skip = SkipLayer(inChannels, outChannels)

    def forward(self, x):
        out = 0
        out = out + self.cb(x)
        out = out + self.skip(x)
        return out


class MyUpsample(nn.Module):
    def __init__(self):
        super(MyUpsample, self).__init__()
        pass

    def forward(self, x):
        return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1),
                                                                              x.size(2)*2, x.size(3)*2)
