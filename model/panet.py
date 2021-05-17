import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rpn import *


class AdaptiveFeaturePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scaled_features, rois):
        for i in range(len(scaled_features) - 1, -1, -1):
            roi = rois * 2 ** i




class PANet(nn.Module):
    def __init__(self):
        super(PANet, self).__init__()

        self.rpn = RPN(in_channel=256, feature_size=28)
        self.backbone = self.rpn.backbone

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


if __name__ == '__main__':
    from torchsummary import summary
    model = Backbone().cuda()
    summary(model, (3, 224, 224))




























