from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1_norm():
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.ind_dict = {}
    
    def __call__(self, *args: nn.Module, **kwargs: None) -> Any:
    #def __call__(self, model: nn.Module) -> Any:
        self.L1_norm_executor(args[0])
        return self.ind_dict
        
    def L1_norm_executor(self, model, name='', ):
        for name1, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    sort_index = self.channel_L1_norm(layer.weight)
                    name_ = name + name1
                    self.ind_dict[name_] = sort_index
            else:
                name += name1 + "."
                self.L1_norm_executor(layer, name)

    def channel_L1_norm(self, weight: torch.tensor) -> torch.tensor:
        out_channels = weight.shape[0]
        importances = []
        for i_c in range(out_channels):
            channel_weight = weight.detach()[i_c]
            importance = torch.norm(channel_weight)
            importances.append(importance.view(1))
        importance = torch.cat(importances)
        sort_index = torch.argsort(importance, descending=True)

        return sort_index


def channels_vulnerability_gain(net, target_layer, trainloader, batch_size, out_no, device):
    torch.cuda.empty_cache()
    if isinstance(target_layer, nn.Conv2d):
        vulnerability = torch.zeros(target_layer.in_channels, device=device)
        for data in trainloader:
            x, _ = data[0].to(device), data[1].to(device)

            #run a forward pass                
            outputs = net(x)
            _, predicted = torch.max(outputs, 1)

            #HarDNN's gain implementation
            one_hots = F.one_hot(predicted, num_classes=out_no)
            out_difs = torch.unsqueeze(torch.sum(outputs * one_hots, 1), 1) - outputs
            out_difs_sq = torch.pow(out_difs, 2)
            vul_in_channel_total = torch.zeros(target_layer.in_channels, device=device)

            for cls in range(out_no):
                target_layer.weight.requires_grad = True
                target_layer.weight.retain_grad()
                loss = out_difs[:, cls]
                loss.backward(torch.ones_like(loss), retain_graph=True)

                grad_sq = torch.pow(target_layer.weight.grad.data, 2)
                out_difs_sq_cp = torch.clone(out_difs_sq)
                out_difs_sq_cp[out_difs_sq_cp == 0] = 1
                
                vul_in_channel = torch.zeros(target_layer.in_channels, device=device)
                for img in range(batch_size):
                    vul_ch = torch.div(grad_sq, out_difs_sq_cp[img, cls])
                    vul_in_channel += torch.sum(torch.sum(torch.sum(vul_ch, 3), 2), 0)
                
                vul_in_channel = vul_in_channel / batch_size
                vul_in_channel_total += vul_in_channel

                target_layer.weight.grad.zero_()
                
                #free memory
                del target_layer.weight.grad
                del loss
                del grad_sq
                del out_difs_sq_cp
                del vul_in_channel
                torch.cuda.empty_cache()
            
            vul_in_channel_total = torch.div(vul_in_channel_total, out_no)
            vulnerability += vul_in_channel_total
            
            break
        
        vulnerability = torch.div(vulnerability, torch.max(vulnerability))

    #free memory
    del net
    del target_layer
    del trainloader
    torch.cuda.empty_cache()

    return vulnerability