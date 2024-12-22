'''
source code is reused and modified from: https://github.com/hamidmousavi0/reliable-relu-toolbox/blob/master/src/rrelu/search_bound/ranger.py
'''

import torch
import torch.nn as nn
from typing import Dict, Union, Any
from torch.utils.data import DataLoader
import logging

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook


def relu_hooks(model: nn.Module,
               name: str=''):
    for name1, layer in model.named_children():
        if list(layer.children()) == []:
            if isinstance(layer, nn.ReLU):
                name_ = name + name1
                layer.register_forward_hook(get_activation(name_)) 
        else:
            name += name1 + "."
            relu_hooks(layer, name)
             
def Ranger_thresholds(model:nn.Module, 
                      dataloader: DataLoader, 
                      device: Union[torch.device, str],
                      logger: logging.Logger) -> torch.tensor:
    model.eval()
    thresholds = {}
    thresholds_tmp = {}
    relu_hooks(model)
    init_flag = 1

    for data, _ in dataloader:
        data = data.to(device)
        _ = model(data)
        for key, val in activations.items():
            thresholds_tmp[key] = torch.max(val) 
            if init_flag:
                thresholds[key] = torch.tensor(0) 

        for key, val in activations.items():
            total_max = thresholds[key]
            curr_max = thresholds_tmp[key]
            total_max = curr_max if curr_max > total_max else total_max
            thresholds[key] = total_max 
        
        init_flag = 0

    logger.info("thresholds are derived based on Ranger")
    return thresholds