import torch as th
from torch import nn


class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, channel):
        super(tdBatchNorm, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        self.weight.data.mul_(0.5)

    def forward(self, x):
        B, T, *spatial_dims = x.shape
        out = super().forward(x.reshape(B * T, *spatial_dims))
        BT, *spatial_dims = out.shape
        out = out.view(B, T, *spatial_dims).contiguous()
        return out

class tdBatchNormBTDimFuse(nn.BatchNorm2d):
    def __init__(self, channel):
        super(tdBatchNormBTDimFuse, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        self.weight.data.mul_(0.5)

    def forward(self, x):
        out = super().forward(x)
        return out


if __name__ == "__main__":
    input_ = th.randn(4, 3, 2, 224, 224)
    input_ = input_.reshape(-1, 2, 224, 224)
    bn = tdBatchNormBTDimFuse(2)
    output = bn(input_)
    print(output.shape)
