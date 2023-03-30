import torch
import torch.nn as nn

from libft2.multichannel_layernorm import MultichannelLayerNorm
from libft2.squared_relu import SquaredReLU

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features, kernel_size=3, padding=1, downsample=False, stride=(2,1)):
        super(ResBlock, self).__init__()

        self.activate = SquaredReLU()
        self.norm = MultichannelLayerNorm(in_channels, features)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride if downsample else 1, bias=False)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride if downsample else 1, bias=False) if in_channels != out_channels or downsample else nn.Identity()

    def forward(self, x):
        h = self.conv2(self.activate(self.conv1(self.norm(x))))
        x = self.identity(x) + h

        return x