from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from torch.utils.data import DataLoader
from DeepVigor_analysis import DeepVigor_analysis
import fault_simulation
import binary_converter
import copy
import run_mode_utils
from pathlib import Path
import os


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
    

class DeepVigor():
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.model = args[0]        #nn.Module
        self.inputs = args[1]       #torch.tensor
        self.out_classes = args[2]  #int
        self.device = args[3]       #Union[torch.device, str]
        self.logger = args[4]       #logging.Logger
        self.pruning_ratio = args[5]
        self.hardening_ratio = args[6]
        
        self.layers_output_count_dict = {}
        self.input_activations_dict = {}
        self.ind_dict = {}
        self.vf_files_dir = self.logger.handlers[0].baseFilename.split("log")[0] + "/../deepvigor"

    def __call__(self, *args, **kwds):
        if not os.path.exists(self.vf_files_dir):
            os.makedirs(self.vf_files_dir)
        self.DeepVigor_executor(self.model)
        return self.ind_dict

    def __get_neurons_count(self, name):
        def hook(model, input, output):
            self.layers_output_count_dict[name] = output[0].numel()
            self.input_activations_dict[name] = input[0]
        return hook

    def DeepVigor_executor(self, model, layer_name=''):
        for name1, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    name = layer_name + name1
                    cvf_file_dir = f"{self.vf_files_dir}/channels-VF-{name}-{self.pruning_ratio}-{self.hardening_ratio}.txt"
                    if not Path(cvf_file_dir).is_file():
                        critical_cvf = self.DeepVigor_analysis(layer, name)
                        self.save_cvf(critical_cvf, cvf_file_dir)
                        sort_index = torch.argsort(critical_cvf, descending=True)
                    else:
                        critical_cvf = self.load_cvf(layer.out_channels, cvf_file_dir)
                        sort_index = torch.argsort(critical_cvf, descending=True)
                    self.ind_dict[name] = sort_index

                    torch.cuda.empty_cache()
            
            elif isinstance(layer, nn.Module) and hasattr(layer, 'conv1') and hasattr(layer, 'conv2'):  #BasicBlocks in ResNet
                name_ = layer_name + name1 + ".conv1"
                cvf_file_dir = f"{self.vf_files_dir}/channels-VF-{name}-{self.pruning_ratio}-{self.hardening_ratio}.txt"
                if not Path(cvf_file_dir).is_file():
                    critical_cvf = self.DeepVigor_analysis(layer.conv1, name_)
                    self.save_cvf(critical_cvf, cvf_file_dir)
                    sort_index = torch.argsort(critical_cvf, descending=True)
                else:
                    critical_cvf = self.load_cvf(layer.conv1.out_channels, cvf_file_dir)
                    sort_index = torch.argsort(critical_cvf, descending=True)
                self.ind_dict[name_] = sort_index

                name_ = layer_name + name1 + ".conv2"
                cvf_file_dir = f"{self.vf_files_dir}/channels-VF-{name}-{self.pruning_ratio}-{self.hardening_ratio}.txt"
                if not Path(cvf_file_dir).is_file():
                    critical_cvf = self.DeepVigor_analysis(layer.conv2, name_)
                    self.save_cvf(critical_cvf, cvf_file_dir)
                    sort_index = torch.argsort(critical_cvf, descending=True)
                else:
                    critical_cvf = self.load_cvf(layer.conv2.out_channels, cvf_file_dir)
                    sort_index = torch.argsort(critical_cvf, descending=True)
                self.ind_dict[name_] = sort_index
                
                if layer.downsample:
                    for name_tmp, sub_layer in layer.downsample.named_children():
                        if isinstance(sub_layer, nn.Conv2d):
                            name_ = layer_name + name1 + ".downsample." + name_tmp
                            cvf_file_dir = f"{self.vf_files_dir}/channels-VF-{name}-{self.pruning_ratio}-{self.hardening_ratio}.txt"
                            if not Path(cvf_file_dir).is_file():
                                critical_cvf = self.DeepVigor_analysis(sub_layer, name_)
                                self.save_cvf(critical_cvf, cvf_file_dir)
                                sort_index = torch.argsort(critical_cvf, descending=True)
                            else:
                                critical_cvf = self.load_cvf(sub_layer.out_channels, cvf_file_dir)
                                sort_index = torch.argsort(critical_cvf, descending=True)
                            self.ind_dict[name_] = sort_index
            
            else:
                layer_name += name1 + "."
                self.DeepVigor_executor(layer, layer_name)


    def DeepVigor_analysis(self,
                           layer: nn.Module, 
                           name: str) -> torch.tensor:
        handle = layer.register_forward_hook(self.__get_neurons_count(name))
        _ = self.model(self.inputs)
        handle.remove()

        deepvigor = DeepVigor_analysis(self.model, self.device)
        x_pad = F.pad(self.input_activations_dict[name], (layer.padding[0], layer.padding[0], layer.padding[0], layer.padding[0]))
        layer_info_set = {"stride": layer.stride[0], "kernel_size": layer.kernel_size[0], "neurons_in_layer": self.layers_output_count_dict[name], 
                        "batch_size": self.inputs.size(0), "out_channel": layer.out_channels, 
                        "layer_inputs": x_pad, "layer_weights": layer.weight}
        
        # deriving channel vulnerability factors (CVF)
        critical_cvf = deepvigor.channels_vulnerability_factor(self.inputs, layer, layer_info_set, self.out_classes)    

        self.logger.info(f"Derived and saved the vulnerability factors for layer {name}")
        
        del deepvigor
        del x_pad

        return critical_cvf

    
    def save_cvf(self,
                 critical_cvf: torch.tensor,
                 cvf_file_dir: str) -> None:
        cvf_file = open(cvf_file_dir, 'w')
        for i in range(critical_cvf.size(0)):
            cvf_file.write(str(critical_cvf[i].item()) + "\n")
        cvf_file.close()

    
    def load_cvf(self,
                 out_channels: int,
                 cvf_file_dir: str) -> torch.tensor:
        critical_cvf = torch.zeros(out_channels, device=self.device)
        cvf_file = open(cvf_file_dir, 'r')
        i = 0
        for line in cvf_file:
            critical_cvf[i] = float(line)
            i += 1
        cvf_file.close()
        self.logger.info(f"Loaded the vulnerability factors from {cvf_file_dir}")

        return critical_cvf

class channel_FI():
    def __init__(self, *args, **kwds):
        self.model = args[0]
        self.dataloader = args[1]
        self.device = args[2]
        self.logger = args[3]

        self.sort_index_dict = {}

    def __call__(self, *args, **kwds):
        self.channel_ranking_FI(self.model)
        self.logger.info("applied FI to all channels")
        return self.sort_index_dict

    def channel_ranking_FI(self,
                           model: nn.Module,
                           name: str = "") -> None:

        with torch.no_grad():
            for name1, layer in model.named_children():
                if list(layer.children()) == []:
                    if isinstance(layer, nn.Conv2d): #or isinstance(layer, hardening.HardenedConv2d):
                        torch.cuda.empty_cache()
                        channel_bits = torch.numel(layer.weight[0]) * 32
                        repetition_count = int(channel_bits / (1 + (0.0001 * ((channel_bits - 1) / 0.96))))
                        repetition_count = 2000 if repetition_count > 2000 else repetition_count

                        channels_vf_list = torch.zeros(layer.weight.size(0), device=self.device)
                        for ch in range(layer.weight.size(0)):      #iterate over out_channels
                            total_faulty_accuracy = 0
                            for _ in range(repetition_count):
                                weight_size = layer.weight.size()
                                w1 = torch.randint(weight_size[1], (1,), device=self.device).item()
                                w2 = torch.randint(weight_size[2], (1,), device=self.device).item()
                                w3 = torch.randint(weight_size[3], (1,), device=self.device).item()
                                random_bits = torch.randint(32, (1,), device=self.device).item()        #32-bit data
                                layer.weight.requires_grad = False
                                golden_weight_cp = copy.deepcopy(layer.weight[ch, w1, w2, w3])
                                weight_bit = binary_converter.float2bit(layer.weight[ch, w1, w2, w3].unsqueeze(0), device=self.device)
                                weight_bit[0][random_bits] = torch.logical_xor(weight_bit[0][random_bits], torch.tensor(1)).float() 
                                layer.weight[ch, w1, w2, w3] = nn.Parameter(binary_converter.bit2float(weight_bit, device=self.device)[0])

                                _, faulty_accuracy = fault_simulation.model_evaluation(model, self.dataloader, self.device)
                                total_faulty_accuracy += faulty_accuracy

                                layer.weight[ch, w1, w2, w3] = copy.deepcopy(golden_weight_cp)
                                
                                del golden_weight_cp
                                del weight_bit

                            channels_vf_list[ch] = 1 - total_faulty_accuracy / repetition_count

                        name_ = name + name1
                        self.sort_index_dict[name_] = torch.argsort(channels_vf_list, descending=True)
                        self.logger.info(f"Channel ranking is performed using FI for layer {name_} with {repetition_count} repetition of FI")
                        run_mode_utils.save_dict(self.sort_index_dict, "channel-FI-temp", self.logger)

                else:
                    name += name1 + '.'
                    self.channel_ranking_FI(layer, name)