from torch import nn
from layers.tdbn import tdBatchNorm
from spikingjelly.activation_based.neuron import LIFNode


def BasicBlock_CBLM(in_channels, out_channels, with_max_pooling=True):
    """
    Args:
        with_max_pooling: 是否使用最大池化层
        in_channels: 输入通道
        out_channels: 输出通道

    Returns: 一个顺序容器，包含CBLM模块
        上采样通道，下采样分辨率
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
        tdBatchNorm(out_channels),
        LIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, step_mode='m', backend="cuda", store_v_seq=False),
        nn.MaxPool2d(kernel_size=2, stride=2) if with_max_pooling else nn.Identity()
    )


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        """
        define backbone and detection head here
        """
        pass

    def forward(self, x):
        pass

