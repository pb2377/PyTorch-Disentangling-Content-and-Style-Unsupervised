import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class LeakyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, vector_out=False, bias=True):
        super(LeakyBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.lrelu = nn.LeakyReLU()
        self.vector_out = vector_out
        if vector_out:
            self.instn = nn.InstanceNorm2d(1, affine=True)
        else:
            self.instn = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        if self.vector_out:
            x = self.conv(self.lrelu(x))
            # assert torch.equal(x, x.view(x.size(0), 1, -1, 1).view_as(x))
            return self.instn(x.view(x.size(0), 1, -1, 1)).view_as(x)
        else:
            return self.instn(self.conv(self.lrelu(x)))


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, bias=True):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, bias=bias)
        self.relu = nn.ReLU()
        self.instn = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.instn(self.deconv(self.relu(x)))


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x