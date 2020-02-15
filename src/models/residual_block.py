import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, dim, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3x3(dim, dim, stride)
        self.bn1 = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(dim, dim)
        self.bn2 = nn.BatchNorm3d(dim)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


if __name__ == '__main__':
  x = torch.Tensor(1, 128, 8, 32, 32)
  print("Input: ", x.size())
  channlel = x.size()[1]
  block = ResidualBlock(channlel)
  out = block(x)
  print("Output: ", out.size())
  