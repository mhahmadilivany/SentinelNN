import torch
import torch.nn as nn
import pruning
import copy

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
                              model: nn.Module) -> None:
        
        for _, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    self.conv_layers.append(layer)
                elif isinstance(layer, nn.BatchNorm2d):
                    self.bn_layers.append(layer)
                elif isinstance(layer, nn.Linear):
                    self.fc_layers.append(layer)

            else:
                self._get_separated_layers(layer)
        
    def _reset_params(self):
        self.conv_count = 0
        self.conv_layers = []
        self.bn_layers = []
        self.fc_layers = []

        

    '''
    iterates over the model,
    applies homogeneous pruning to conv and fc layers separately,
    returns the pruned model 
    '''
    def homogeneous_prune(self) -> nn.Module:
        self._get_separated_layers(self.model)
        assert len(self.conv_layers) != 0
        assert len(self.conv_layers) == len(self.bn_layers)
        class_count = self.fc_layers[-1].out_features
        with torch.no_grad():
            for i in range(len(self.conv_layers)):
                conv_layer = self.conv_layers[i]
                new_out_channel = conv_layer.out_channels - int(conv_layer.out_channels * self.conv_pruning_ratio)
                print(conv_layer, new_out_channel)
                conv_layer.weight.set_(conv_layer.weight.detach()[:new_out_channel])
                conv_layer.bias.set_(conv_layer.bias.detach()[:new_out_channel])
                conv_layer.out_channels = new_out_channel

                bn_layer = self.bn_layers[i]

                bn_layer.weight.set_(bn_layer.weight.detach()[:new_out_channel])
                bn_layer.bias.set_(bn_layer.bias.detach()[:new_out_channel])
                bn_layer.running_mean.set_(bn_layer.running_mean.detach()[:new_out_channel])
                bn_layer.running_var.set_(bn_layer.running_var.detach()[:new_out_channel])
                bn_layer.num_features = new_out_channel
                
                if i < len(self.conv_layers) - 1:
                    next_conv = self.conv_layers[i+1]
                    next_conv.weight.set_(next_conv.weight.detach()[:, :new_out_channel])
                    next_conv.in_channels = new_out_channel
            
            new_features_count = new_out_channel
            for i in range(len(self.fc_layers)):
                fc_layer = self.fc_layers[i]
                print(fc_layer, new_features_count)
                fc_layer.in_features = new_features_count
                fc_layer.weight.set_(fc_layer.weight.detach()[: , :new_features_count])

                if fc_layer.out_features != class_count:
                    new_features_count = fc_layer.out_features - int(fc_layer.out_features * self.fc_pruning_ratio)
                    fc_layer.weight.set_(fc_layer.weight.detach()[:new_features_count, :])
                    fc_layer.bias.set_(fc_layer.bias.detach()[:new_features_count])
                    fc_layer.out_features = new_features_count


        return self.model