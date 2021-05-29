import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torchvision.ops import *

from dataset.penn_fudan_dataset import *
from utils.pytorch_util import *
from model.conv import *


if __name__ == '__main__':
    def test():
        return 1, 2, 3
    a, b, c = test()
    l = [*test()]
    print(l)
