import torch
import torch.nn as nn
import binary_converter
import copy
import hardening
import torch.nn as nn
from typing import Dict, Union, Any
from torch.utils.data import DataLoader
import logging

class fault_counter():
    def __init__(self):
        self.fault_count = 0

# injecting faults into weights of a Conv2d layer
def conv_FI(layer: nn.Conv2d, 
           faults_count: int,
           device: Union[torch.device, str]) -> None:
    
    weight_size = layer.weight.size()
    rand_w0 = torch.randint(weight_size[0], (faults_count,), device=device)
    rand_w1 = torch.randint(weight_size[1], (faults_count,), device=device)
    rand_w2 = torch.randint(weight_size[2], (faults_count,), device=device)
    rand_w3 = torch.randint(weight_size[3], (faults_count,), device=device)
    random_bits = torch.randint(32, (faults_count,))        #32-bit data
    layer.weight.requires_grad = False
    for fault in range(faults_count):
        w0, w1, w2, w3 = rand_w0[fault].item(), rand_w1[fault].item(), rand_w2[fault].item(), rand_w3[fault].item()
        weight_bit = binary_converter.float2bit(layer.weight[w0, w1, w2, w3].unsqueeze(0), device=device) 
        weight_bit[0][random_bits[fault]] = torch.logical_xor(weight_bit[0][random_bits[fault]], torch.tensor(1)).float() 
        layer.weight[w0, w1, w2, w3] = nn.Parameter(binary_converter.bit2float(weight_bit, device=device)[0])    


# injecting faults into weights of a Linear layer
def linear_FI(layer: nn.Linear, 
             faults_count: int, 
             device: Union[torch.device, str]) -> None:
    
    rand_w0 = torch.randint(layer.out_features, (faults_count,), device=device)
    rand_w1 = torch.randint(layer.in_features, (faults_count,), device=device)
    random_bits = torch.randint(32, (faults_count,))        #32-bit data
    
    layer.weight.requires_grad = False
    for fault in range(faults_count):
        w0, w1 = rand_w0[fault].item(), rand_w1[fault].item()
        weight_bit = binary_converter.float2bit(layer.weight[w0, w1].unsqueeze(0), device=device) 
        weight_bit[0][random_bits[fault]] = torch.logical_xor(weight_bit[0][random_bits[fault]], torch.tensor(1)).float() 
        layer.weight[w0, w1] = nn.Parameter(binary_converter.bit2float(weight_bit, device=device)[0])


def weights_FI(model: nn.Module, 
               BER: float, 
               device: Union[torch.device, str],
               counter: fault_counter) -> tuple[nn.Module]:
    for _, layer in model.named_children():
        if list(layer.children()) == []:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, hardening.HardenedConv2d):
                weight_faults_count = int(BER * layer.weight.numel() * 32)      #32-bit data
                counter.fault_count += weight_faults_count
                conv_FI(layer, weight_faults_count, device)

            elif isinstance(layer, nn.Linear):
                weight_faults_count = int(BER * layer.weight.numel() * 32)      #32-bit data
                counter.fault_count += weight_faults_count
                linear_FI(layer, weight_faults_count, device)

        else:
            weights_FI(layer, BER, device, counter)

    return model


def model_evaluation(model: nn.Module, 
                            dataloader: DataLoader, 
                            device: Union[torch.device, str],
                            ) -> Union[torch.tensor, float]:
    model.eval()
    golden_accuracy = 0
    total_data = 0
    for data in dataloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model.forward(images)
        _, golden_predicted = torch.max(outputs.data, 1)
        golden_accuracy += (golden_predicted == labels).sum()
        total_data += labels.size(0)
        break

    golden_accuracy = golden_accuracy / total_data
    return golden_predicted, golden_accuracy


def faulty_model_evaluation(model: nn.Module, 
                            dataloader: DataLoader, 
                            classes_count: int, 
                            golden_predicted: torch.tensor, 
                            device: Union[torch.device, str],
                            ) -> tuple[int, float, float, float]:
    model.eval()
    faulty_accuracy = 0
    total_data = 0
    for data in dataloader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = model.forward(images)
        _, faulty_predicted = torch.max(outputs.data, 1)
        total_data += labels.size(0)
        
        #removing NaN outputs from the classification
        faulty_predicted = torch.nan_to_num(faulty_predicted, nan=classes_count+1, posinf=classes_count+1, neginf=classes_count+1)
        #faulty_values = torch.nan_to_num(faulty_values, nan=10000, posinf=10000, neginf=10000)
        faulty_accuracy += (faulty_predicted == labels).sum()

        #fault classification
        DUE = torch.sum(faulty_predicted == (classes_count+1)).item() / (labels.size(0))
        SDC_critical = torch.sum(faulty_predicted != golden_predicted).item() / (labels.size(0)) - DUE
        SDC_non_critical = torch.sum(faulty_predicted == golden_predicted).item() / (labels.size(0))
        break

    faulty_accuracy = faulty_accuracy / total_data
    return faulty_accuracy, DUE, SDC_critical, SDC_non_critical




    #return channels_vf