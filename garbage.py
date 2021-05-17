import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from dataset.penn_fudan_dataset import *


if __name__ == '__main__':
    for i in range(3, -1, -1):
        print(i)