import torch.nn as nn


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


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.acti = nn.ReLU(True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.acti(x)

        return x


if __name__ == '__main__':
    import torch

    a = torch.ones(1, 3, 143, 143)
    deconv = nn.ConvTranspose2d(3, 8, 3, 2, 1, 1)
    b = deconv(a)
    print(b.shape)






















