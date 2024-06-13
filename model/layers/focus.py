import torch
from torch import nn
from spikingjelly.activation_based.neuron import LIFNode
from tdbn import tdBatchNormBTDimFuse

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "lif":
        module = LIFNode(tau=2.0, detach_reset=True, backend='torch', step_mode='m', store_v_seq=False)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm/tdBatchNorm -> silu/leaky relu/lif block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        # self.bn = nn.BatchNorm2d(out_channels)
        self.bn = tdBatchNormBTDimFuse(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    """
    Focus width and height information into channel space.
    """
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="lif"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        # shape of x (b,t,c,h,w) -> y(b, t, 4c, h/2, w/2)
        B, T, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x).reshape(B, T, -1, H // 2, W // 2)


if __name__ == "__main__":
    input_ = torch.randn(4, 3, 2, 224, 224)
    model = Focus(2, 32)
    output = model(input_)
    print(output.shape)
