from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1_norm():
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.ind_dict = {}
    
    def __call__(self, *args: nn.Module, **kwargs: None) -> Any:
        self.L1_norm_executor(args[0])
        return self.ind_dict
        
    def L1_norm_executor(self, 
                         model: nn.Module, 
                         name: str = '') -> None:
        for name1, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    sort_index = self.channel_L1_norm(layer.weight)
                    name_ = name + name1
                    self.ind_dict[name_] = sort_index
            
            elif isinstance(layer, nn.Module) and hasattr(layer, 'conv1') and hasattr(layer, 'conv2'):  #BasicBlocks in ResNet
                name_ = name + name1 + ".conv1"
                sort_index = self.channel_L1_norm(layer.conv1.weight)
                self.ind_dict[name_] = sort_index

                name_ = name + name1 + ".conv2"
                sort_index = self.channel_L1_norm(layer.conv2.weight)
                self.ind_dict[name_] = sort_index

                if layer.downsample:
                    for name_tmp, sub_layer in layer.downsample.named_children():
                        if isinstance(sub_layer, nn.Conv2d):
                            name_ = name + name1 + ".downsample." + name_tmp
                            sort_index = self.channel_L1_norm(sub_layer.weight)
                            self.ind_dict[name_] = sort_index

            else:
                name += name1 + "."
                self.L1_norm_executor(layer, name)

    def channel_L1_norm(self, 
                        weight: torch.tensor) -> torch.tensor:
        out_channels = weight.shape[0]
        importances = []
        for i_c in range(out_channels):
            channel_weight = weight.detach()[i_c]
            importance = torch.norm(channel_weight)
            importances.append(importance.view(1))
        importance = torch.cat(importances)
        sort_index = torch.argsort(importance, descending=True)

        return sort_index


class vulnerability_gain():
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.model = args[0]
        self.dataloader = args[1]
        self.out_classes = args[2]
        self.device = args[3]
        self.ind_dict = {}

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.gain_norm_executor(self.model)
        return self.ind_dict

    def gain_norm_executor(self, 
                           model: nn.Module, 
                           name: str = '') -> None:
        for name1, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    sort_index = self.channels_vulnerability(self.model, layer)
                    name_ = name + name1
                    self.ind_dict[name_] = sort_index
            
            elif isinstance(layer, nn.Module) and hasattr(layer, 'conv1') and hasattr(layer, 'conv2'):  #BasicBlocks in ResNet
                sort_index = self.channels_vulnerability(self.model, layer.conv1)
                name_ = name + name1 + ".conv1"
                self.ind_dict[name_] = sort_index

                sort_index = self.channels_vulnerability(self.model, layer.conv2)
                name_ = name + name1 + ".conv2"
                self.ind_dict[name_] = sort_index

                if layer.downsample:
                    for name_tmp, sub_layer in layer.downsample.named_children():
                        if isinstance(sub_layer, nn.Conv2d):
                            sort_index = self.channels_vulnerability(self.model, sub_layer)
                            name_ = name + name1 + ".downsample." + name_tmp
                            self.ind_dict[name_] = sort_index
            
            else:
                name += name1 + "."
                self.gain_norm_executor(layer, name)

    def channels_vulnerability(self, 
                               model: nn.Module, 
                               target_layer: nn.Module) -> torch.tensor:
        
        torch.cuda.empty_cache()
        trainloader = self.dataloader
        batch_size = trainloader.batch_size
        out_classes = self.out_classes
        vulnerability = torch.zeros(target_layer.out_channels, device=self.device)
        for data in trainloader:
            x = data[0].to(self.device)        
            outputs = model(x).to(self.device)
            _, predicted = torch.max(outputs, 1)

            #HarDNN's gain implementation
            one_hots = F.one_hot(predicted, num_classes=out_classes)
            out_difs = torch.unsqueeze(torch.sum(outputs * one_hots, 1), 1) - outputs
            out_difs_sq = torch.pow(out_difs, 2)
            vul_out_channel_total = torch.zeros(target_layer.out_channels, device=self.device)

            for cls in range(out_classes):
                target_layer.weight.requires_grad = True
                target_layer.weight.retain_grad()
                loss = out_difs[:, cls]
                loss.backward(torch.ones_like(loss), retain_graph=True)

                grad_sq = torch.pow(target_layer.weight.grad.data, 2)
                out_difs_sq_cp = torch.clone(out_difs_sq)
                out_difs_sq_cp[out_difs_sq_cp == 0] = 1
                
                vul_out_channel = torch.zeros(target_layer.out_channels, device=self.device)
                for img in range(batch_size):
                    vul_ch = torch.div(grad_sq, out_difs_sq_cp[img, cls])
                    vul_out_channel += torch.sum(torch.sum(torch.sum(vul_ch, 3), 2), 1)
                
                vul_out_channel = vul_out_channel / batch_size
                vul_out_channel_total += vul_out_channel

                target_layer.weight.grad.zero_()
                
                #free memory
                del target_layer.weight.grad
                del loss
                del grad_sq
                del out_difs_sq_cp
                del vul_out_channel
                torch.cuda.empty_cache()
            
            vul_out_channel_total += torch.div(vul_out_channel_total, out_classes)
            vulnerability += vul_out_channel_total
            
            break
        
        vulnerability = torch.div(vulnerability, torch.max(vulnerability))

        #free memory
        del model
        del target_layer
        del trainloader
        torch.cuda.empty_cache()
        sort_index = torch.argsort(vulnerability, descending=True)

        return sort_index
    

class Salience():
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.salience = torch.ones(args[1], device=args[2])     #num_classes: int
        self.device = args[2]                                   #device: 'cuda', 'cpu'
        self.ind_dict = {}
    
    def __call__(self, *args: nn.Module, **kwargs: None) -> Any:
        self.Salience_executor(args[0])                         #model
        return self.ind_dict
        
    def Salience_executor(self, model, name='', ) -> None:
        for name1, layer in reversed(list(model.named_children())):
            if list(layer.children()) == []:
                if isinstance(layer, nn.Linear):
                    name_ = name + name1
                    self.ind_dict[name_] = self.neurons_salience(layer.weight)

                elif isinstance(layer, nn.Conv2d):
                    name_ = name + name1
                    self.ind_dict[name_] = torch.argsort(self.salience, descending=True)
                    _ = self.channels_salience(layer.weight)

            elif isinstance(layer, nn.Module) and hasattr(layer, 'conv1') and hasattr(layer, 'conv2'):  #BasicBlocks in ResNet
                if layer.downsample:
                    for name_tmp, sub_layer in layer.downsample.named_children():
                        if isinstance(sub_layer, nn.Conv2d):
                            name_ = name + name1 + ".downsample." + name_tmp
                            self.ind_dict[name_] = torch.argsort(self.salience, descending=True)
                
                name_ = name + name1 + ".conv2"
                self.ind_dict[name_] = torch.argsort(self.salience, descending=True)
                _ = self.channels_salience(layer.conv2.weight)

                name_ = name + name1 + ".conv1"
                self.ind_dict[name_] = torch.argsort(self.salience, descending=True)
                _ = self.channels_salience(layer.conv1.weight)

            else:
                name += name1 + "."
                self.Salience_executor(layer, name)

    def neurons_salience(self,
                          weight: torch.tensor) -> torch.tensor:
        
        self.salience = torch.sum(self.salience.unsqueeze(1) * torch.abs(weight.detach()), dim=0)
        sort_index = torch.argsort(self.salience, descending=True)

        return sort_index
    
    def channels_salience(self, 
                        weight: torch.tensor) -> torch.tensor:
        
        out_channels = weight.shape[0]
        new_salience = torch.zeros(weight.shape[1], device=self.device)
        
        for i_c in range(out_channels):
            channel_weight = torch.abs(weight.detach()[i_c])
            new_salience += torch.sum(torch.sum(torch.sum(self.salience[i_c] * channel_weight, dim=2), dim=1), dim=0)
            
        self.salience = new_salience
        sort_index = torch.argsort(self.salience, descending=True)

        return sort_index