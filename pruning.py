from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import models_utils
from typing import Union
import logging


def out_channel_sorting(conv_layer: nn.Module, 
                    sort_index: torch.tensor) -> None:
    conv_layer.weight.copy_(
        torch.index_select(conv_layer.weight.detach(), 0, sort_index))

    #conv_layer.bias.copy_(
    #    torch.index_select(conv_layer.bias.detach(), 0, sort_index))


def in_channel_sorting(conv_layer: nn.Module, 
                    sort_index: torch.tensor) -> None:
    conv_layer.weight.copy_(
        torch.index_select(conv_layer.weight.detach(), 1, sort_index))


def batchnorm_sorting(bn_layer: nn.Module,
                      sort_index: torch.tensor) -> None:
    for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
        tensor_to_apply = getattr(bn_layer, tensor_name)
        tensor_to_apply.copy_(
            torch.index_select(tensor_to_apply.detach(), 0, sort_index))
    

def fine_tune(model: nn.Module,
              trainloader: DataLoader,
              testloader: DataLoader,
              eopchs: int,
              device: Union[torch.device, str],
              logger: logging.Logger,
              pruning_ratio: float,
              importance_command: str) -> nn.Module:
    
    logger.info("fine-tuning the pruned model")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eopchs)
    criterion = nn.CrossEntropyLoss()
    log_dir = logger.handlers[0].baseFilename.split("log")[0]

    best_accuracy = 0
    best_epoch = -1
    for epoch in range(eopchs):
        models_utils.train(model, trainloader, criterion, optimizer, scheduler, device=device)
        accuracy = models_utils.evaluate(model, testloader, device=device)
        is_best = accuracy >= best_accuracy
        if is_best:
            best_accuracy = accuracy
            best_epoch = epoch
            
            torch.save(model.state_dict(), f'{log_dir}/../pruned_model-{importance_command}-{pruning_ratio}.pth')
        logger.info(f"epoch {epoch}, accuracy: {accuracy}%")
        
    logger.info(f"model saved with the best achieved accuracy, i.e., epoch {best_epoch}, accuracy: {best_accuracy}%")

    return model