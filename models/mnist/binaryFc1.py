import torch
import torch.nn as nn

from .utils import BinaryLinear

class fc1(nn.Module):

    def __init__(self, num_classes=10):
        super(fc1, self).__init__()
        self.classifier = nn.Sequential(
            BinaryLinear(28*28, 1000),
            nn.ReLU(inplace=True),
            BinaryLinear(1000, 500),
            nn.ReLU(inplace=True),
            BinaryLinear(500, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    