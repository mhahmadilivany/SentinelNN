import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import DataLoader

from typing import Union


class HardenedConv2d(nn.Conv2d):
    def __init__(self, hardening_ratio, *args, **kwargs):
        self.hardening_ratio = hardening_ratio
        super(HardenedConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        #input channels remain intact
        #applying normal forward which leads to more output channels
        #reducing the size of output channels while correction, leading to the expected out_channels in next layer

        return super(HardenedConv2d, self).forward(x)