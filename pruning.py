from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import models_utils
from typing import Union


def out_channel_sorting(conv_layer: nn.Module, 
                    sort_index: torch.tensor) -> nn.Module:
    conv_layer.weight.copy_(
        torch.index_select(conv_layer.weight.detach(), 0, sort_index))

    return conv_layer

def in_channel_sorting(conv_layer: nn.Module, 
                    sort_index: torch.tensor) -> nn.Module:
    conv_layer.weight.copy_(
        torch.index_select(conv_layer.weight.detach(), 1, sort_index))

    return conv_layer

def batchnorm_sorting(bn_layer: nn.Module,
                      sort_index: torch.tensor) -> nn.Module:
    for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
        tensor_to_apply = getattr(bn_layer, tensor_name)
        tensor_to_apply.copy_(
            torch.index_select(tensor_to_apply.detach(), 0, sort_index))
    
    return bn_layer

def fc_sorting(fc_layer: nn.Module,
               sort_index: torch.tensor) -> nn.Module:
    fc_layer.weight.copy_(
        torch.index_select(fc_layer.weight.detach(), 0, sort_index))
    
    return fc_layer


def fine_tune(model: nn.Module,
              trainloader: DataLoader,
              testloader: DataLoader,
              eopchs: int,
              device: Union[torch.device, str]) -> nn.Module:
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eopchs)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0
    best_epoch = 0
    for epoch in range(eopchs):
        models_utils.train(model, trainloader, criterion, optimizer, scheduler, device=device)
        accuracy = models_utils.evaluate(model, testloader, device=device)
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            best_epoch = epoch
        print(epoch, accuracy)
        
    print(best_accuracy, best_epoch)

    return model