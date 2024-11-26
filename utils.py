import torch
import torch.nn as nn
import pruning
import copy
import sys
from typing import Callable, Dict, Any
import importance_analysis

class AnalysisHandler():
    def __init__(self) -> None:
        self.functions: Dict[str, Callable] = {}

    def register(self,
                 command: str,
                 func: Callable) -> None:
        self.functions[command] = func

    def execute(self, 
                command: str, *args, **kwargs) -> Any:
        if command in self.functions:
            print(f"Executing '{command}'")
            obj = self.functions[command](*args, **kwargs)
            return obj(*args, **kwargs)
        else:
            raise ValueError(f"Unknown command: {command}. Available commands: {list(self.functions.keys())}")
        


class prune_utils():
    def __init__(self, 
                 model: nn.Module,
                 pruning_method: str) -> None:
        
        if not isinstance(pruning_method, str):
            raise TypeError(f"Expected 'pruning_method' to be a string, got {type(pruning_method).__name__}")
        
        self.model = model
        self.pruning_method = pruning_method    #hm: homogeneous, ht: heterogeneous
        self.conv_count = 0
        self.conv_layers = []
        self.bn_layers = []
        self.fc_layers = []
        self.importance = {}


    def set_pruning_ratios(self, 
                           pruning_params: list) -> None:
        
        if not isinstance(pruning_params, list):
            raise TypeError(f"Expected 'pruning_params' to be a list, got {type(pruning_params).__name__}")
        
        if self.pruning_method == "hm":
            self.conv_pruning_ratio = pruning_params[0]
            self.fc_pruning_ratio = pruning_params[1]
        
        elif self.pruning_method == "ht":
            self.conv_count = 0
            self._conv_layer_counter(self.model)
            self.conv_pruning_ratio = pruning_params[:self.conv_count]
            self.fc_pruning_ratio = pruning_params[self.conv_count:]

    
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
                        handler: AnalysisHandler,
                        command: str="l1-norm") -> nn.Module:
        
        if command == "l1-norm":
            sort_index_conv_dict = handler.execute(command, model, ...)   #importance_analysis.channel_L1_norm(conv_layer.weight)
            print(sort_index_conv_dict)
        #elif command == "vul-gain":
        #    sort_index_conv = importance_analysis.channels_vulnerability_gain(model, data, ...)

        self._reset_params()
        self._get_separated_layers(model)
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
    


