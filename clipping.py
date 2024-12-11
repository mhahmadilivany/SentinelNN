'''
source code is reused and modified from: https://github.com/hamidmousavi0/reliable-relu-toolbox/blob/master/src/rrelu/search_bound/ranger.py
'''

import torch
import torch.nn as nn
from typing import Dict, Any
from torch.utils.data import DataLoader
from typing import Union
#import os
import logging

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook


def relu_hooks(model:nn.Module):
    for name, layer in model.named_children():
        if list(layer.children()) == []:
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(get_activation(name)) 
        else:
            relu_hooks(layer)
             
def Ranger_thresholds(model:nn.Module, 
                      dataloader: DataLoader, 
                      device: Union[torch.device, str],
                      logger: logging.Logger) -> torch.tensor:
    model.eval()
    thresholds = {}
    relu_hooks(model)
    for data, _ in dataloader:
        data = data.to(device)
        _ = model(data)
        
        for key, val in activations.items():
            thresholds[key] = val
        
        total_max = 0
        for key, val in activations.items():
            curr_max = torch.max(activations[key])
            total_max = curr_max if curr_max > total_max else total_max
            thresholds[key] = total_max 
        
        #break 
    logger.info("thresholds are derived based on Ranger")
    return thresholds