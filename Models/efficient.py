import torch
import torch.nn as nn
from Models.conv import conv1x1, conv3x3


def efficientL2(**kwargs):
    return EfficientNet()
