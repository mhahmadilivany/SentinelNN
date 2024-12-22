import torch
import torch.nn as nn
import pruning
from typing import Dict, Any
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

            elif isinstance(layer, nn.Module) and hasattr(layer, 'conv1') and hasattr(layer, 'conv2'):  #BasicBlocks in ResNet
                name_ = name + name1
                self.conv_layers.append([name_, layer])     #self.basic_block can be considered

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

    
    def _conv_prune(self, 
                    conv_layer: nn.Conv2d, 
                    next_conv: Union[nn.Conv2d, nn.Linear]) -> None:
        new_out_channel = conv_layer.out_channels - int(conv_layer.out_channels * self.conv_pruning_ratio)
                        
        conv_layer.weight.set_(conv_layer.weight.detach()[:new_out_channel])
        if conv_layer.bias is not None:
            conv_layer.bias.set_(conv_layer.bias.detach()[:new_out_channel])
        conv_layer.out_channels = new_out_channel
        
        if next_conv is not None:
            if isinstance(next_conv, nn.Conv2d):
                next_conv.weight.set_(next_conv.weight.detach()[:, :new_out_channel])
                next_conv.in_channels = new_out_channel
            elif isinstance(next_conv, nn.Linear):
                next_conv.weight.set_(next_conv.weight.detach()[:, :new_out_channel])
                next_conv.in_features = new_out_channel
                            

    def _bn_prune(self, bn_layer: nn.BatchNorm2d) -> None:
        new_out_channel = bn_layer.num_features - int(bn_layer.num_features * self.conv_pruning_ratio)
        
        bn_layer.weight.set_(bn_layer.weight.detach()[:new_out_channel])
        bn_layer.bias.set_(bn_layer.bias.detach()[:new_out_channel])
        bn_layer.running_mean.set_(bn_layer.running_mean.detach()[:new_out_channel])
        bn_layer.running_var.set_(bn_layer.running_var.detach()[:new_out_channel])
        bn_layer.num_features = new_out_channel


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
        assert len(self.fc_layers) != 0

        with torch.no_grad():
            for i in range(len(self.conv_layers)):      # - 1):
                conv_layer = self.conv_layers[i][1]
                if i + 1 < len(self.conv_layers):
                    next_conv = self.conv_layers[i+1][1]
                else:
                    next_conv = self.fc_layers[0]
                
                if "BasicBlock" in type(conv_layer).__name__:
                    self._conv_prune(conv_layer.conv1, conv_layer.conv2)
                    self._bn_prune(conv_layer.bn1)

                    if "BasicBlock" in type(next_conv).__name__:
                        self._conv_prune(conv_layer.conv2, next_conv.conv1)
                        self._bn_prune(conv_layer.bn2)
                    else:
                        self._conv_prune(conv_layer.conv2, next_conv)
                        self._bn_prune(conv_layer.bn2)
                        
                    
                    if conv_layer.downsample:
                        for _, sub_layer in conv_layer.downsample.named_children():
                            if isinstance(sub_layer, nn.Conv2d):
                                sub_layer.weight.set_(sub_layer.weight.detach()[:, :conv_layer.conv1.in_channels])
                                sub_layer.in_channels = conv_layer.conv1.in_channels
                                self._conv_prune(sub_layer, None)
                            elif isinstance(sub_layer, nn.BatchNorm2d):
                                self._bn_prune(sub_layer)

                else:
                    if "BasicBlock" in type(next_conv).__name__:
                        self._conv_prune(conv_layer, next_conv.conv1)
                        self._bn_prune(self.bn_layers[i])
                    else:
                        self._conv_prune(conv_layer, next_conv)
                        self._bn_prune(self.bn_layers[i])
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
            self.fc_layers.reverse()

        assert len(self.conv_layers) != 0
        assert len(self.fc_layers) != 0
        
        with torch.no_grad():
            for i in range(len(self.conv_layers)):  # - 1):
                conv_name = self.conv_layers[i][0]
                conv_layer = self.conv_layers[i][1]
                if i + 1 < len(self.conv_layers):
                    next_conv = self.conv_layers[i+1][1]
                else:
                    next_conv = self.fc_layers[0]

                if "BasicBlock" in type(conv_layer).__name__:                #for resnet
                    layer_name = conv_name + ".conv1"
                    sort_index_conv = sort_index_conv_dict[layer_name]
                    pruning.out_channel_sorting(conv_layer.conv1, sort_index_conv)
                    pruning.batchnorm_sorting(conv_layer.bn1, sort_index_conv)

                    pruning.in_channel_sorting(conv_layer.conv2, sort_index_conv)                    

                    if conv_layer.downsample:
                        layer_name = conv_name + ".conv2"
                        sort_index_conv = sort_index_conv_dict[layer_name]
                        pruning.out_channel_sorting(conv_layer.conv2, sort_index_conv)
                        pruning.batchnorm_sorting(conv_layer.bn2, sort_index_conv)
                        
                        for _, sub_layer in conv_layer.downsample.named_children():
                            if isinstance(sub_layer, nn.Conv2d):
                                pruning.in_channel_sorting(sub_layer, last_sort_index_conv)
                                pruning.out_channel_sorting(sub_layer, sort_index_conv)
                            elif isinstance(sub_layer, nn.BatchNorm2d):
                                pruning.batchnorm_sorting(sub_layer, sort_index_conv)

                        if "BasicBlock" in type(next_conv).__name__:
                            pruning.in_channel_sorting(next_conv.conv1, sort_index_conv)
                        else:
                            pruning.in_channel_sorting(next_conv, sort_index_conv)

                        last_sort_index_conv = sort_index_conv

                    else:
                        pruning.out_channel_sorting(conv_layer.conv2, last_sort_index_conv)
                        pruning.batchnorm_sorting(conv_layer.bn2, last_sort_index_conv)

                        if "BasicBlock" in type(next_conv).__name__:
                            pruning.in_channel_sorting(next_conv.conv1, last_sort_index_conv)
                        else:
                            pruning.in_channel_sorting(next_conv, last_sort_index_conv)

                else:
                    sort_index_conv = sort_index_conv_dict[conv_name]
                    pruning.out_channel_sorting(conv_layer, sort_index_conv)
                    pruning.batchnorm_sorting(self.bn_layers[i], sort_index_conv)
                    
                    if "BasicBlock" in type(next_conv).__name__:
                        pruning.in_channel_sorting(next_conv.conv1, sort_index_conv)
                    else:
                        pruning.in_channel_sorting(next_conv, sort_index_conv)
                    
                    last_sort_index_conv = sort_index_conv
            
        return model
    

class hardening_utils():
    def __init__(self, 
                 hardening_ratio: float,
                 clipping_command: str="ranger") -> None:
        self.hardening_ratio = hardening_ratio
        self.clipping = clipping_command
        self.relu_thresholds = {}

 
    def conv_replacement(self, model: nn.Module):
        if self.hardening_ratio != 0:
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
    def hardening_conv(self, layer: nn.Conv2d):
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
                if layer.bias is not None:
                    new_bias[i] = layer.bias[i]
                    new_bias[duplication_count + i] = layer.bias[i]
            new_weight[2*duplication_count:] = layer.weight[duplication_count:]
            if layer.bias is not None:
                new_bias[2*duplication_count:] = layer.bias[duplication_count:]

            layer.weight = nn.Parameter(new_weight)
            if layer.bias is not None:
                layer.bias = nn.Parameter(new_bias)

        return layer


    def thresholds_extraction(self, 
                              model: nn.Module,
                              handler: handlers.ClippingHandler,
                              clipping_command: str,
                              dataloader: DataLoader,
                              device: Union[torch.device, str],
                              logger: logging.Logger) -> None:
        self.relu_thresholds = {}
        if clipping_command == "ranger":
            self.relu_thresholds = handler.execute(clipping_command, model, dataloader, device, logger)


    def relu_replacement(self, 
                         model: nn.Module,
                         name: str = '') -> nn.Module:
        for name1, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.ReLU):
                    hardened_relu = hardening.RangerReLU(
                        inplace=layer.inplace
                    )
                    name_ = name + name1
                    hardened_relu = self.hardening_relu(hardened_relu, self.relu_thresholds[name_].item())
                    
                    # Replace the module in the model
                    setattr(model, name1, hardened_relu)

            else:
                name += name1 + "."
                self.relu_replacement(layer, name)
        
        return model
    

    def hardening_relu(self, relu: nn.ReLU, threshold: float):
        hardened_relu = hardening.RangerReLU(inplace=relu.inplace)
        hardened_relu.clipping_threshold = threshold
        return hardened_relu

