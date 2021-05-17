import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.acti = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.acti(x)

        return x


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.encoder = nn.Sequential(
            Conv(3, 32, 3, 2, 1),
            Conv(32, 64, 3, 2, 1),
            Conv(64, 128, 3, 2, 1),
            Conv(128, 256, 3, 2, 1)
        )
        self.lateral = nn.Sequential(
            Conv(128, 256, 1),
            Conv(64, 256, 1),
            Conv(32, 256, 1)
        )

    def forward(self, x):
        f2 = self.encoder[0](x)
        f3 = self.encoder[1](f2)
        f4 = self.encoder[2](f3)
        f5 = self.encoder[3](f4)

        P5 = f5
        P4 = F.interpolate(P5, scale_factor=2) + self.lateral[0](f4)
        P3 = F.interpolate(P4, scale_factor=2) + self.lateral[1](f3)
        P2 = F.interpolate(P3, scale_factor=2) + self.lateral[2](f2)

        return P2, P3, P4, P5


class BottomUpNetwork(nn.Module):
    def __init__(self):
        super(BottomUpNetwork, self).__init__()
        self.encoder = nn.Sequential(
            Conv(256, 256, 3, 2, 1),
            Conv(256, 256, 3, 2, 1),
            Conv(256, 256, 3, 2, 1)
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

        return N2, N3, N4, N5