from copy import copy

import torch
from spikingjelly.activation_based.neuron import LIFNode
from torch import nn

from tdbn import tdBatchNormBTDimFuse


class BottleNeck(nn.Module):
    def __init__(self, channels, with_residual=True):
        super().__init__()
        self.with_residual = with_residual
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels, padding=0, kernel_size=1, stride=1)
        self.tdBN_1 = tdBatchNormBTDimFuse(channels)
        self.lif_1 = LIFNode(tau=2.0, detach_reset=True, backend='torch', step_mode='m', store_v_seq=False)

        self.conv_2 = nn.Conv2d(in_channels=channels, out_channels=channels, padding=1, kernel_size=3, stride=1)
        self.tdBN_2 = tdBatchNormBTDimFuse(channels)
        self.lif_2 = LIFNode(tau=2.0, detach_reset=True, backend='torch', step_mode='m', store_v_seq=False)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        input_x = copy(x)
        x = self.conv_1(x)
        x = self.tdBN_1(x).reshape(B, T, C, H, W).permute(1, 0, 2, 3, 4)
        x = self.lif_1(x).reshape(B*T, C, H, W)

        x = self.conv_2(x)
        x = self.tdBN_2(x).reshape(B, T, C, H, W).permute(1, 0, 2, 3, 4)
        x = self.lif_2(x).reshape(B*T, C, H, W)

        return (input_x + x).reshape(B, T, C, H, W) if self.with_residual else x.reshape(B, T, C, H, W)


class SPPBottleNeck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels // 2, padding=0, kernel_size=1, stride=1)
        self.tdBN_1 = tdBatchNormBTDimFuse(channels // 2)
        self.lif_1 = LIFNode(tau=2.0, detach_reset=True, backend='torch', step_mode='m', store_v_seq=False)

        self.pool_1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool_2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool_3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        self.conv_2 = nn.Conv2d(in_channels=channels*2, out_channels=channels, padding=0, kernel_size=1, stride=1)
        self.tdBN_2 = tdBatchNormBTDimFuse(channels)
        self.lif_2 = LIFNode(tau=2.0, detach_reset=True, backend='torch', step_mode='m', store_v_seq=False)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = self.conv_1(x)
        x = self.tdBN_1(x).reshape(B, T, C//2, H, W).permute(1, 0, 2, 3, 4)
        x = self.lif_1(x).reshape(B*T, C//2, H, W)

        x_1 = self.pool_1(x)
        x_2 = self.pool_2(x)
        x_3 = self.pool_3(x)

        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        x = self.conv_2(x)
        x = self.tdBN_2(x).reshape(B, T, C, H, W).permute(1, 0, 2, 3, 4)
        x = self.lif_2(x).permute(1, 0, 2, 3, 4)
        return x


if __name__ == "__main__":
    input_ = torch.randn(4, 3, 2, 224, 224)
    bottleneck = BottleNeck(channels=2, with_residual=False)
    output = bottleneck(input_)
    print(output.shape)

    spp_bottleneck = SPPBottleNeck(channels=2)
    output = spp_bottleneck(input_)
    print(output.shape)

