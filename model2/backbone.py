import torch.nn as nn

from torchvision.models import vgg16


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vgg16()

    def forward(self, x):
        x = self.model(x)

        return x


def _backbone():
    return Backbone()


if __name__ == '__main__':
    from torchsummary import summary
    model = _backbone().cuda()
    summary(model, (3, 448, 448))