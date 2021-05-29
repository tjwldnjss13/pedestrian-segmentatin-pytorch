import torch
import torch.nn as nn
import torch.nn.functional as F

from model.conv import *


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        if pool:
            self.conv1 = Conv(in_channels, out_channels, 3, 2, 1)
            self.conv_skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = Conv(in_channels, out_channels, 3, 1, 1)
            self.conv_skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        skip = self.conv_skip(skip)
        x += skip

        return x


def _residual_block(in_channels, out_channels, pool=False, num_repeat=1):
    return nn.Sequential(*[ResidualBlock(in_channels, out_channels, pool) for _ in range(num_repeat)])


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        # self.encoder = nn.Sequential(
        #     Conv(3, 32, 3, 2, 1),
        #     Conv(32, 64, 3, 2, 1),
        #     Conv(64, 128, 3, 2, 1),
        #     Conv(128, 256, 3, 2, 1)
        # )
        self.encoder1 = nn.Sequential(
            Conv(3, 64, 3, 2, 1),
            _residual_block(64, 64, num_repeat=3)
        )
        self.encoder2 = nn.Sequential(
            _residual_block(64, 128, pool=True),
            _residual_block(128, 128, num_repeat=3)
        )
        self.encoder3 = nn.Sequential(
            _residual_block(128, 256, pool=True),
            _residual_block(256, 256, num_repeat=5)
        )
        self.encoder4 = nn.Sequential(
            _residual_block(256, 512, pool=True),
            _residual_block(512, 512, num_repeat=2)
        )
        self.lateral = nn.Sequential(
            Conv(256, 512, 1),
            Conv(128, 512, 1),
            Conv(64, 512, 1)
        )

    def forward(self, x):
        # f2 = self.encoder[0](x)
        # f3 = self.encoder[1](f2)
        # f4 = self.encoder[2](f3)
        # f5 = self.encoder[3](f4)
        f2 = self.encoder1(x)
        f3 = self.encoder2(f2)
        f4 = self.encoder3(f3)
        f5 = self.encoder4(f4)

        P5 = f5
        P4 = F.interpolate(P5, scale_factor=2) + self.lateral[0](f4)
        P3 = F.interpolate(P4, scale_factor=2) + self.lateral[1](f3)
        P2 = F.interpolate(P3, scale_factor=2) + self.lateral[2](f2)

        return P2, P3, P4, P5


class BottomUpNetwork(nn.Module):
    def __init__(self):
        super(BottomUpNetwork, self).__init__()
        self.encoder = nn.Sequential(
            Conv(512, 512, 3, 2, 1),
            Conv(512, 512, 3, 2, 1),
            Conv(512, 512, 3, 2, 1)
        )

    def forward(self, P2, P3, P4, P5):
        N2 = P2
        N3 = self.encoder[0](N2) + P3
        N4 = self.encoder[1](N3) + P4
        N5 = self.encoder[2](N4) + P5

        return N2, N3, N4, N5


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.fpn = FPN()
        self.bottom_up = BottomUpNetwork()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        P2, P3, P4, P5 = self.fpn(x)
        N2, N3, N4, N5 = self.bottom_up(P2, P3, P4, P5)
        # N2, N3, N4, N5 = self.bottom_up(*self.fpn(x))

        # return N2, N3, N4, N5
        return N5


if __name__ == '__main__':
    import numpy as np

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def get_model_size_in_mb(model):
        mods = list(model.modules())
        for i in range(1, len(mods)):
            m = mods[i]
            p = list(m.parameters())
            sizes = []
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        total_bits = 0
        for i in range(len(sizes)):
            s = sizes[i]
            bits = np.prod(np.array(s)) * bits
            total_bits += bits

        return total_bits

    from torchsummary import summary

    backbone = Backbone().cuda()
    fpn = FPN().cuda()
    summary(backbone, (3, 448, 448))
    print(get_n_params(backbone))
































