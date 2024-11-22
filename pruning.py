from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# class importance_analysis():
#     def __init__(self, importance_method) -> None:
#         self.importance_method = importance_method

#     def __call__(self, *args: Any, **kwds: Any) -> Any:
#         #args.self.func(args.weight)
#         pass


def input_channel_L1_norm(weight: torch.tensor) -> torch.tensor:
    out_channels = weight.shape[0]
    importances = []
    for i_c in range(out_channels):
        #channel_weight = weight.detach()[:, i_c]
        channel_weight = weight.detach()[i_c]
        importance = torch.norm(channel_weight)
        importances.append(importance.view(1))
    importance = torch.cat(importances)
    sort_index = torch.argsort(importance, descending=True)

    return sort_index


def channel_sorting(conv_layer: nn.Module, 
                    sort_index: torch.tensor) -> nn.Module:
    conv_layer.weight.copy_(
        torch.index_select(conv_layer.weight.detach(), 0, sort_index))

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