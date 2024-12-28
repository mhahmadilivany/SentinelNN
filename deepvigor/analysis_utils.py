import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import DeepVigor
import time
import os.path



class analysis_utils():
    def __init__(self, net, network_name, inputs, dataset_name, batch_size, classes_count, log_file_name, log_direction, device, channel_sampling_ratio=None, neuron_sampling=None):
        self.net = net
        self.network_name = network_name
        self.inputs = inputs
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.classes_count = classes_count
        self.log_file_name = log_file_name
        self.log_direction = log_direction 
        self.device = device
        
        self.layers_output_count_dict = {}
        self.input_activations_dict = {}
        self.weights_dict = {}
        self.__get_neurons_layers(self.net)
        self.total_analysis_time = 0
        self.total_analyzed_neurons = 0
        self.LVF_dict = {}

        if channel_sampling_ratio:
            self.channel_sampling_ratio = channel_sampling_ratio
            self.layers_count = len(self.layers_output_count_dict)
            self.channel_errors = {} 
        
        if neuron_sampling:
            self.neuron_sampling = neuron_sampling

    def __get_neurons_count(self, name):
        def hook(model, input, output):
            self.layers_output_count_dict[name] = output[0].numel()
            self.input_activations_dict[name] = input[0]
            self.weights_dict[name] = model.weight
        return hook

    def __get_neurons_layers(self, model, name=''):
        for name1, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    name_ = name + name1
                    layer.register_forward_hook(self.__get_neurons_count(name_))
            else:
                name += name1 + "."     
                self.__get_neurons_layers(layer, name)

    
    def full_weights_analysis(self, model, layer_name=''):
        for name1, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    name = layer_name + name1
                    handle = layer.register_forward_hook(self.__get_neurons_count(name))
                    _ = self.net(self.inputs)
                    handle.remove()

                    deep_vigor = DeepVigor.DeepVigor(self.net, self.network_name, self.device)
                    name = layer_name + name1
                    x_pad = F.pad(self.input_activations_dict[name], (layer.padding[0], layer.padding[0], layer.padding[0], layer.padding[0]))
                    layer_info_set = {"stride": layer.stride[0], "kernel_size": layer.kernel_size[0], "neurons_in_layer": self.layers_output_count_dict[name], 
                                    "batch_size": self.batch_size, "out_channel": layer.out_channels, 
                                    "layer_inputs": x_pad, "layer_weights": layer.weight}

                    file_name_weight = self.log_direction + "/full-weights-non_critical-nvf-" + self.network_name + "-" + self.dataset_name + "-" + name + ".txt"
                    tmp = time.time()

                    non_critical_cvf_weights, analysis_counters_channels = deep_vigor.all_weights_vulnerability_factor(self.inputs, layer, layer_info_set, self.classes_count)
                    
                    #log the analysis execution:
                    log_file = open(self.log_file_name, 'a')
                    log_file.write("analysis results for layer: " + name + "\n")
                    log_file.write("analysis results based on analyzing weights, layer: " + name + "\n")
                    log_file.write("total channels: " + str(analysis_counters_channels[0].item()) + "\n")
                    log_file.write("total number of forwards: " + str(analysis_counters_channels[11].item()) + "\n")
                    log_file.write("LVF based on weights: " + str(1 - (torch.sum(non_critical_cvf_weights).item() / layer.out_channels)) + "\n")
                    log_file.write("layer analysis time: " + str(time.time() - tmp))
                    log_file.write("\n\n\n")
                    log_file.close()

                    #save the results, each line represents CVF for a channel
                    noncritical_nvf_file = open(file_name_weight, 'w')
                    for i in range(layer.out_channels):
                        noncritical_nvf_file.write(str(1 - non_critical_cvf_weights[i].item()) + "\n")
                    noncritical_nvf_file.close()
                
                    del deep_vigor
                    del non_critical_cvf_weights
                    del analysis_counters_channels
                    torch.cuda.empty_cache()

            else:
                layer_name += name1 + "."
                self.full_weights_analysis(layer, layer_name)


    def full_activations_analysis(self, model, layer_name=''):
        for name1, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    name = layer_name + name1
                    if os.path.exists(self.network_name + "-" + self.dataset_name + "/full-activations-non_critical-nvf-" + self.network_name + "-" + self.dataset_name + "-" + name + ".txt") == False:
                        handle = layer.register_forward_hook(self.__get_neurons_count(name))
                        _ = self.net(self.inputs)
                        handle.remove()

                        deep_vigor = DeepVigor.DeepVigor(self.net, self.network_name, self.device)
                        name = layer_name + name1
                        x_pad = F.pad(self.input_activations_dict[name], (layer.padding[0], layer.padding[0], layer.padding[0], layer.padding[0]))
                        layer_info_set = {"stride": layer.stride[0], "kernel_size": layer.kernel_size[0], "neurons_in_layer": self.layers_output_count_dict[name], 
                                        "batch_size": self.batch_size, "out_channel": layer.out_channels, 
                                        "layer_inputs": x_pad, "layer_weights": layer.weight}
                        
                        file_name_act = self.log_direction + "/full-activations-non_critical-nvf-" + self.network_name + "-" + self.dataset_name + "-" + name + ".txt"
                        tmp = time.time()

                        non_critical_nvf_activations, analysis_counters_neurons = deep_vigor.all_activations_vulnerability_factor(self.inputs, layer, layer_info_set, self.classes_count, file_name_act)
                        
                        log_file = open(self.log_file_name, 'a')
                        log_file.write("analysis results based on analyzing activations, layer: " + name + "\n")
                        log_file.write("total neurons: " + str(analysis_counters_neurons[0].item()) + "\n")
                        log_file.write("total number of forwards: " + str(analysis_counters_neurons[11].item()) + "\n")
                        log_file.write("LVF based on activations: " + str(1 - (torch.sum(non_critical_nvf_activations).item() / analysis_counters_neurons[0].item())) + "\n")
                        log_file.write("layer analysis time: " + str(time.time() - tmp))
                        log_file.write("\n\n")
                        log_file.close()

                        #save neuron vulnerability factor in each line
                        noncritical_nvf_file = open(file_name_act, 'w')
                        for i in range(self.layers_output_count_dict[name]):
                            noncritical_nvf_file.write(str(1 - non_critical_nvf_activations[i].item()) + "\n")
                        noncritical_nvf_file.close()
                    
                        del deep_vigor
                        del non_critical_nvf_activations
                        del analysis_counters_neurons
                    torch.cuda.empty_cache()

            else:
                layer_name += name1 + "."
                self.full_activations_analysis(layer, layer_name)

    
    def sampling_analysis_act(self, model, layer_name=''):
        for name1, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    name = layer_name + name1

                    handle = layer.register_forward_hook(self.__get_neurons_count(name))
                    _ = self.net(self.inputs)
                    handle.remove()

                    deep_vigor = DeepVigor.DeepVigor(self.net, self.network_name, self.device)
                    name = layer_name + name1
                    x_pad = F.pad(self.input_activations_dict[name], (layer.padding[0], layer.padding[0], layer.padding[0], layer.padding[0]))
                    layer_info_set = {"stride": layer.stride[0], "kernel_size": layer.kernel_size[0], "neurons_in_layer": self.layers_output_count_dict[name], 
                                    "batch_size": self.batch_size, "out_channel": layer.out_channels, 
                                    "layer_inputs": x_pad, "layer_weights": layer.weight}
                    
                    tmp = time.time()
                    noncritical_nvf_list, analysis_counters_neurons, sampled_CVF = deep_vigor.sampling_activations_vulnerability_factor(self.inputs, layer, layer_info_set, self.channel_sampling_ratio, self.classes_count)
                    sampled_LVF = torch.sum(noncritical_nvf_list) / analysis_counters_neurons[0]
                    analysis_time = time.time() - tmp
                    self.total_analysis_time += analysis_time
                    self.total_analyzed_neurons += analysis_counters_neurons[0]
                    
                    log_file = open(self.log_file_name, 'a')
                    log_file.write("analysis results based on analyzing sampled activations, layer: " + name + "\n")
                    log_file.write("total neurons: " + str(analysis_counters_neurons[0].item()) + "\n")
                    log_file.write("total number of forwards: " + str(analysis_counters_neurons[11].item()) + "\n")
                    log_file.write("LVF based on activations: " + str(1 - sampled_LVF.item()) + "\n")
                    log_file.write("layer analysis time: " + str(analysis_time))
                    log_file.write("\n\n")
                    log_file.close()

            else:
                layer_name += name1 + "."
                self.sampling_analysis_act(layer, layer_name)


    def sampling_analysis_weight(self, model, layer_name=''):
        for name1, layer in model.named_children():
            if list(layer.children()) == []:
                if isinstance(layer, nn.Conv2d):
                    name = layer_name + name1
                    handle = layer.register_forward_hook(self.__get_neurons_count(name))
                    _ = self.net(self.inputs)
                    handle.remove()

                    deep_vigor = DeepVigor.DeepVigor(self.net, self.network_name, self.device)
                    name = layer_name + name1
                    x_pad = F.pad(self.input_activations_dict[name], (layer.padding[0], layer.padding[0], layer.padding[0], layer.padding[0]))
                    layer_info_set = {"stride": layer.stride[0], "kernel_size": layer.kernel_size[0], "neurons_in_layer": self.layers_output_count_dict[name], 
                                    "batch_size": self.batch_size, "out_channel": layer.out_channels, 
                                    "layer_inputs": x_pad, "layer_weights": layer.weight}
                    
                    tmp = time.time()
                    noncritical_nvf_list, analysis_counters_weights, sampled_LVF = deep_vigor.sampling_weights_vulnerability_factor(self.inputs, layer, layer_info_set, self.channel_sampling_ratio, self.classes_count)
                    analysis_time = time.time() - tmp
                    self.total_analysis_time += analysis_time
                    self.total_analyzed_neurons += analysis_counters_weights[0]

                    log_file = open(self.log_file_name, 'a')
                    log_file.write("analysis results based on analyzing sampled weights, layer: " + name + "\n")
                    log_file.write("total neurons: " + str(analysis_counters_weights[0].item()) + "\n")
                    log_file.write("total number of forwards: " + str(analysis_counters_weights[11].item()) + "\n")
                    log_file.write("LVF based on weights: " + str(1 - sampled_LVF.item()) + "\n")
                    log_file.write("layer analysis time: " + str(analysis_time))
                    log_file.write("\n\n")
                    log_file.close()

            else:
                layer_name += name1 + "."
                self.sampling_analysis_weight(layer, layer_name)
