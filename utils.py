import torch
import torch.nn as nn
import pruning
from typing import Callable, Dict, Any
from torch.utils.data import DataLoader
from typing import Union
import hardening
import handlers
import logging


class prune_utils():
    def __init__(self, 
                 model: nn.Module,
                 trainloader: DataLoader,
                 classes_count: int, 
                 pruning_method: str,
                 device: Union[torch.device, str] = torch.device("cuda")) -> None:
        
        if not isinstance(pruning_method, str):
            raise TypeError(f"Expected 'pruning_method' to be a string, got {type(pruning_method).__name__}")
        
        self.model = model
        self.trainloader = trainloader
        self.classes_count = classes_count
        self.pruning_method = pruning_method    #hm: homogeneous, ht: heterogeneous
        self.conv_count = 0
        self.conv_layers = []
        self.bn_layers = []
        self.fc_layers = []
        self.importance = {}
        self.device = device


    def set_pruning_ratios(self, 
                           pruning_params: Union[float, list]) -> None:
        
        if not isinstance(pruning_params, list) and not isinstance(pruning_params, float):
            raise TypeError(f"Expected 'pruning_params' to be a float, got {type(pruning_params).__name__}")
        
        if self.pruning_method == "hm":
            self.conv_pruning_ratio = pruning_params
            #self.fc_pruning_ratio = pruning_params[1]
        
        elif self.pruning_method == "ht":
            self.conv_count = 0
            self._conv_layer_counter(self.model)
            assert self.conv_count != 0
            self.conv_pruning_ratio = pruning_params[:self.conv_count]
            #self.fc_pruning_ratio = pruning_params[self.conv_count:]

    
    def _conv_layer_counter(self,
                            model: nn.Module) -> None:
        
        for _, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    self.conv_count += 1
            else:
                self._conv_layer_counter(layer)
            
    
    def _get_separated_layers(self, 
                              model: nn.Module,
                              name: str = '') -> None:
        
        for name1, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    name_ = name + name1
                    self.conv_layers.append([name_, layer])
                elif isinstance(layer, nn.BatchNorm2d):
                    self.bn_layers.append(layer)
                elif isinstance(layer, nn.Linear):
                    self.fc_layers.append(layer)

            else:
                name += name1 + "."
                self._get_separated_layers(layer, name)
        
    def _get_separated_layers_reversed(self, 
                              model: nn.Module,
                              name: str = '') -> None:
        
        for name1, layer in reversed(list(model.named_children())):
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    name_ = name + name1
                    self.conv_layers.append([name_, layer])
                elif isinstance(layer, nn.BatchNorm2d):
                    self.bn_layers.append(layer)
                elif isinstance(layer, nn.Linear):
                    self.fc_layers.append(layer)

            else:
                name += name1 + "."
                self._get_separated_layers_reversed(layer, name)

    def _reset_params(self):
        self.conv_count = 0
        self.conv_layers = []
        self.bn_layers = []
        self.fc_layers = []

    '''
    iterates over the model,
    applies homogeneous pruning to conv layers,
    returns the pruned model 
    '''
    def homogeneous_prune(self, 
                          model: nn.Module) -> nn.Module:
        
        self._reset_params()
        self._get_separated_layers(model)
        assert len(self.conv_layers) != 0
        assert len(self.conv_layers) == len(self.bn_layers)

        with torch.no_grad():
            for i in range(len(self.conv_layers) - 1):
                conv_layer = self.conv_layers[i][1]

                new_out_channel = conv_layer.out_channels - int(conv_layer.out_channels * self.conv_pruning_ratio)
                conv_layer.weight.set_(conv_layer.weight.detach()[:new_out_channel])
                conv_layer.bias.set_(conv_layer.bias.detach()[:new_out_channel])
                conv_layer.out_channels = new_out_channel

                bn_layer = self.bn_layers[i]
                bn_layer.weight.set_(bn_layer.weight.detach()[:new_out_channel])
                bn_layer.bias.set_(bn_layer.bias.detach()[:new_out_channel])
                bn_layer.running_mean.set_(bn_layer.running_mean.detach()[:new_out_channel])
                bn_layer.running_var.set_(bn_layer.running_var.detach()[:new_out_channel])
                bn_layer.num_features = new_out_channel

                next_conv = self.conv_layers[i+1][1]
                next_conv.weight.set_(next_conv.weight.detach()[:, :new_out_channel])
                next_conv.in_channels = new_out_channel
            
        return model
    
    '''
    sorts channels in conv layers based on an importance metric
    '''
    def channel_sorting(self, 
                        model: nn.Module,
                        handler: handlers.AnalysisHandler,
                        command: str="l1-norm", 
                        ) -> nn.Module:
        
        self._reset_params()
        if command == "l1-norm":
            sort_index_conv_dict = handler.execute(command, model, ...)  
            self._get_separated_layers(model)
            
        elif command == "vul-gain":
            sort_index_conv_dict = handler.execute(command,model, self.trainloader, self.classes_count, self.device)
            self._get_separated_layers(model)

        elif command == "salience":
            sort_index_conv_dict = handler.execute(command, model, self.classes_count, self.device)
            self._get_separated_layers_reversed(model)
            self.conv_layers.reverse()
            self.bn_layers.reverse()

        assert len(self.conv_layers) != 0
        assert len(self.conv_layers) == len(self.bn_layers)
        
        with torch.no_grad():
            for i in range(len(self.conv_layers) - 1):
                conv_name = self.conv_layers[i][0]
                conv_layer = self.conv_layers[i][1]
                sort_index_conv = sort_index_conv_dict[conv_name]
                
                conv_layer = pruning.out_channel_sorting(conv_layer, sort_index_conv)

                bn_layer = self.bn_layers[i]
                bn_layer = pruning.batchnorm_sorting(bn_layer, sort_index_conv)
                
                next_conv = self.conv_layers[i+1][1]
                conv_layer = pruning.in_channel_sorting(next_conv, sort_index_conv)

        return model
    

class hardening_utils():
    def __init__(self, hardening_ratio: float) -> None:
        self.hardening_ratio = hardening_ratio

    def conv_replacement(self, model):
        for name, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    hardened_layer = hardening.HardenedConv2d(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                        bias=(layer.bias is not None),
                        padding_mode=layer.padding_mode,
                    )
                    hardened_layer.weight.data = layer.weight.data.clone()
                    if layer.bias is not None:
                        hardened_layer.bias.data = layer.bias.data.clone()
                    hardened_layer = self.hardening_conv(hardened_layer)
                    
                    # Replace the module in the model
                    setattr(model, name, hardened_layer)

            else:
                self.conv_replacement(layer)
            
        return model

    # duplicating the weights' output channels based on hardening ratio
    # the model is already sorted, so the first hardening_ratio% should be replicated
    def hardening_conv(self, layer):
        duplication_count = int(layer.out_channels * self.hardening_ratio) 
        duplication_count = 1 if duplication_count == 0 else duplication_count
        layer.duplicated_channels = duplication_count

        new_out_channels = layer.out_channels + duplication_count

        _, b, c, d = layer.weight.size()
        new_weight = torch.zeros(new_out_channels, b, c, d)
        new_bias = torch.zeros(new_out_channels)

        # duplicating the channels
        with torch.no_grad():
            for i in range(duplication_count):
                new_weight[i] = layer.weight[i]
                new_weight[duplication_count + i] = layer.weight[i]
                new_bias[i] = layer.bias[i]
                new_bias[duplication_count + i] = layer.bias[i]
            new_weight[2*duplication_count:] = layer.weight[duplication_count:]
            new_bias[2*duplication_count:] = layer.bias[duplication_count:]

            layer.weight = nn.Parameter(new_weight)
            layer.bias = nn.Parameter(new_bias)
        
        return layer


