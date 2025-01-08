import torch
import torch.nn as nn
import torch.nn.functional as F
import DeepVigor_utils
import math
from typing import Dict, Union


class DeepVigor_analysis():
    def __init__(self, 
                 model: nn.Module, 
                 device: Union[torch.device, str]) -> None:
        self.model = model
        self.device = device
        
    def delta_injection_channel(self, channel, value):
        def hook(model, input, output):
            output[:, channel] += value.unsqueeze(1).unsqueeze(2).expand(output.size(0), output.size(2), output.size(3))
            output.retain_grad()
            self.activation = output
        return hook
    
    def channels_vulnerability_factor(self, 
                                      images: torch.tensor, 
                                      layer: nn.Module, 
                                      layer_info_set: Dict, 
                                      out_no: int) -> torch.tensor:
        features_count = layer_info_set["neurons_in_layer"]
        stride = layer_info_set["stride"]
        kernel_size = layer_info_set["kernel_size"]
        x_pad = layer_info_set["layer_inputs"]
        cnv_weights = layer_info_set["layer_weights"]
        batch_size = layer_info_set["batch_size"]
        out_channel_count = layer_info_set["out_channel"]
        resolution = 10
        inf_represent = 2 ** resolution

        neurons_in_channel = features_count // out_channel_count
        fmap_width = int(math.sqrt(neurons_in_channel))
        
        self.activation = torch.tensor([])
        non_crit_channels = torch.zeros(out_channel_count, device=self.device)
        last_layer_out = self.model.forward(images)
        _, detected_labels = torch.max(last_layer_out, 1)
        one_hots = torch.unsqueeze(F.one_hot(detected_labels, num_classes=out_no), 1)
        neurons_samples = torch.max(torch.tensor([int(torch.log2(torch.tensor([neurons_in_channel]))), 1])).item()
        
        del last_layer_out

        for channel in range(out_channel_count):
            errors_dist_weight_channel = torch.zeros(4 * resolution + 3, device=self.device)
            
            neurons_set = torch.tensor([])
            nrn_counter = 0
            
            while nrn_counter < neurons_samples:
                rand_neuron = torch.randint(neurons_in_channel, (1,)).item()
                if rand_neuron not in neurons_set:
                    nrn_counter += 1
                    neurons_set = torch.cat((neurons_set, torch.tensor([rand_neuron])))

                    neuron_weights = cnv_weights[channel].unsqueeze(0)
                    output_ind_row = rand_neuron // fmap_width
                    output_ind_col = rand_neuron  % fmap_width
                    input_ind_row = output_ind_row * stride
                    input_ind_col = output_ind_col * stride
                    sliced_inputs = x_pad[:, :, 
                        input_ind_row : input_ind_row + kernel_size, 
                        input_ind_col : input_ind_col + kernel_size]
                
                    errors_dist_weight_neuron = DeepVigor_utils.vulnerability_values_space_weight(sliced_inputs, neuron_weights, self.device)
                    errors_dist_weight_channel += errors_dist_weight_neuron
                    
                    del errors_dist_weight_neuron
                    del sliced_inputs
            
            del neurons_set

            #analysis for faulty weights 
            errors_dist_weight_channel = errors_dist_weight_channel / nrn_counter
            VVSS_dict_weights = DeepVigor_utils.creating_VVSS_dict(errors_dist_weight_channel, resolution, self.device)
            
            #finding deltas in negative numbers
            dlt_search_l = -torch.ones(batch_size, device=self.device)
            self.activation = torch.tensor([])
            handle = layer.register_forward_hook(self.delta_injection_channel(channel, dlt_search_l))
            corrupted_out = self.model(images)
            _, corrupted_labels = torch.max(corrupted_out, 1)
            handle.remove()
            
            loss_deepvigor = (torch.sum(torch.sigmoid(torch.unsqueeze(torch.sum(corrupted_out * one_hots, 1), 1) - corrupted_out))) / batch_size
            loss_deepvigor.backward()

            channel_grad = torch.sum(torch.sum(self.activation.grad.data, 3), 2).detach()
            channel_grad[channel_grad != 0] = 1
            grad_bool_map = torch.eq(channel_grad[:, channel], torch.zeros_like(channel_grad[:, channel], device=self.device))

            del self.activation.grad
            del loss_deepvigor
            del corrupted_out
            torch.cuda.empty_cache()

            if torch.sum(channel_grad, 0)[channel] != 0:           #there is some images misclassified by faults
                true_classified = torch.eq(corrupted_labels, detected_labels)
                if torch.sum(true_classified) == batch_size:
                    if VVSS_dict_weights['neg_inf'].size(0) <= 1:               #all images are misclassified by vulnerability_values < -1
                        dlt_search_l = torch.ones_like(dlt_search_l, device=self.device) * (-inf_represent)
                    else:
                        vector_len = VVSS_dict_weights['neg_inf'].size(0)
                        iteration_count = vector_len // 2
                        index_tensor = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count
                        for _ in range(iteration_count, 0, -1):
                            dlt_search_l = VVSS_dict_weights['neg_inf'][index_tensor]
                            handle = layer.register_forward_hook(self.delta_injection_channel(channel, dlt_search_l))
                            corrupted_out = self.model(images)
                            _, corrupted_labels = torch.max(corrupted_out, 1)
                            true_classified = torch.eq(corrupted_labels, detected_labels)
                            index_tensor = torch.logical_not(true_classified) * (index_tensor + 1) + true_classified * (index_tensor - 1)
                            index_tensor[index_tensor >= vector_len] = vector_len - 1
                            index_tensor[index_tensor < 0] = 0
                            handle.remove()

                            del corrupted_out
                            del corrupted_labels
                            del true_classified
                            
                        dlt_search_l = VVSS_dict_weights['neg_inf'][index_tensor]
                        dlt_search_l[grad_bool_map == 1] = -inf_represent
                        torch.cuda.empty_cache()
                
                else:       #images misclassified by vulnerability_values < 0
                    true_classified_init = torch.clone(torch.logical_not(true_classified))
                    vector_len_inf = VVSS_dict_weights['neg_inf'].size(0)
                    iteration_count_inf = vector_len_inf // 2
                    index_tensor_inf = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_inf
                    vector_len_1 = VVSS_dict_weights['neg_1'].size(0)
                    iteration_count_1 = vector_len_1 // 2     #will be bigger value
                    index_tensor_1 = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_1
                    
                    for _ in range(iteration_count_1, 0, -1):
                        dlt_search_l = torch.logical_not(true_classified_init) * VVSS_dict_weights['neg_1'][index_tensor_1] + true_classified_init * VVSS_dict_weights['neg_inf'][index_tensor_inf]
                        handle = layer.register_forward_hook(self.delta_injection_channel(channel, dlt_search_l))
                        corrupted_out = self.model(images)
                        _, corrupted_labels = torch.max(corrupted_out, 1)
                        true_classified = torch.eq(corrupted_labels, detected_labels)
                        
                        index_tensor_inf = true_classified_init * (torch.logical_not(true_classified) * (index_tensor_inf + 1) + true_classified * (index_tensor_inf - 1))
                        index_tensor_1 = torch.logical_not(true_classified_init) * (torch.logical_not(true_classified) * (index_tensor_1 + 1) + true_classified * (index_tensor_1 - 1))
                        index_tensor_inf[index_tensor_inf >= vector_len_inf] = vector_len_inf - 1
                        index_tensor_inf[index_tensor_inf < 0] = 0
                        index_tensor_1[index_tensor_1 >= vector_len_1] = vector_len_1 - 1
                        index_tensor_1[index_tensor_1 < 0] = 0
                        handle.remove()
                        
                        del corrupted_out
                        del corrupted_labels
                        del true_classified

                    dlt_search_l = torch.logical_not(true_classified_init) * VVSS_dict_weights['neg_1'][index_tensor_1] + true_classified_init * VVSS_dict_weights['neg_inf'][index_tensor_inf]
                    dlt_search_l[grad_bool_map == 1] = -inf_represent
                    torch.cuda.empty_cache()

            else:
                #counter for vul < -inf
                dlt_search_l = torch.ones_like(dlt_search_l, device=self.device) * -inf_represent 

            #free memory
            del channel_grad 
            del grad_bool_map
            #del corrupted_out
            torch.cuda.empty_cache()

            #finding deltas in positive numbers
            dlt_search_h = torch.ones(batch_size, device=self.device)
            self.activation = torch.tensor([])
            handle = layer.register_forward_hook(self.delta_injection_channel(channel, dlt_search_h))
            corrupted_out = self.model(images)
            _, corrupted_labels = torch.max(corrupted_out, 1)
            handle.remove()

            loss_deepvigor = (torch.sum(torch.sigmoid(torch.unsqueeze(torch.sum(corrupted_out * one_hots, 1), 1) - corrupted_out))) / batch_size
            loss_deepvigor.backward()

            channel_grad = torch.sum(torch.sum(self.activation.grad.data, 3), 2).detach()
            channel_grad[channel_grad != 0] = 1
            grad_bool_map = torch.eq(channel_grad[:, channel], torch.zeros_like(channel_grad[:, channel], device=self.device))

            del self.activation.grad
            del loss_deepvigor
            del corrupted_out
            torch.cuda.empty_cache()

            if torch.sum(channel_grad, 0)[channel] != 0:          #there is some images misclassified by faults
                true_classified = torch.eq(corrupted_labels, detected_labels)
                if torch.sum(true_classified) == batch_size:        #all images are misclassified by vulnerability_values > 1
                    if VVSS_dict_weights['pos_inf'].size(0) <= 1:
                        dlt_search_h = torch.ones_like(dlt_search_h, device=self.device) * inf_represent 
                    else:
                        vector_len = VVSS_dict_weights['pos_inf'].size(0)
                        iteration_count = vector_len // 2
                        index_tensor = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count
                        for _ in range(iteration_count, 0, -1):
                            dlt_search_h = VVSS_dict_weights['pos_inf'][index_tensor]
                            handle = layer.register_forward_hook(self.delta_injection_channel(channel, dlt_search_h))
                            corrupted_out = self.model(images)
                            _, corrupted_labels = torch.max(corrupted_out, 1)
                            true_classified = torch.eq(corrupted_labels, detected_labels)
                            index_tensor = torch.logical_not(true_classified) * (index_tensor - 1) + true_classified * (index_tensor + 1)
                            index_tensor[index_tensor >= vector_len] = vector_len - 1
                            index_tensor[index_tensor < 0] = 0
                            handle.remove()
                            
                            del true_classified
                            del corrupted_labels
                            del corrupted_out

                        dlt_search_h = VVSS_dict_weights['pos_inf'][index_tensor]
                        dlt_search_h[grad_bool_map == 1] = inf_represent
                        torch.cuda.empty_cache()

                else:
                    vector_len_inf = VVSS_dict_weights['pos_inf'].size(0)
                    vector_len_1 = VVSS_dict_weights['pos_1'].size(0)
                    iteration_count_inf = vector_len_inf // 2
                    iteration_count_1 = vector_len_1 // 2     #will be bigger value
                    index_tensor_inf = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_inf
                    index_tensor_1 = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_1
                    true_classified_init = torch.clone(true_classified)
                    for _ in range(iteration_count_1, 0, -1):
                        dlt_search_h = torch.logical_not(true_classified_init) * VVSS_dict_weights['pos_1'][index_tensor_1] + true_classified_init * VVSS_dict_weights['pos_inf'][index_tensor_inf]
                        handle = layer.register_forward_hook(self.delta_injection_channel(channel, dlt_search_h))
                        corrupted_out = self.model(images)
                        _, corrupted_labels = torch.max(corrupted_out, 1)
                        true_classified = torch.eq(corrupted_labels, detected_labels)
                        index_tensor_inf = true_classified_init * (torch.logical_not(true_classified) * (index_tensor_inf - 1) + true_classified * (index_tensor_inf + 1))
                        index_tensor_1 = torch.logical_not(true_classified_init) * (torch.logical_not(true_classified) * (index_tensor_1 - 1) + true_classified * (index_tensor_1 + 1))
                        index_tensor_inf[index_tensor_inf >= vector_len_inf] = vector_len_inf - 1
                        index_tensor_inf[index_tensor_inf < 0] = 0
                        index_tensor_1[index_tensor_1 >= vector_len_1] = vector_len_1 - 1
                        index_tensor_1[index_tensor_1 < 0] = 0
                        handle.remove()
                        
                        del corrupted_labels
                        del true_classified
                        del corrupted_out


                    dlt_search_h = torch.logical_not(true_classified_init) * VVSS_dict_weights['pos_1'][index_tensor_1] + true_classified_init * VVSS_dict_weights['pos_inf'][index_tensor_inf]
                    dlt_search_h[grad_bool_map == 1] = inf_represent  
                    torch.cuda.empty_cache()

            else:
                dlt_search_h = torch.ones_like(dlt_search_h, device=self.device) * inf_represent  


            #NVF calculation
            negative_vulnerability_powers = torch.zeros_like(dlt_search_l, device=self.device)
            negative_vulnerability_powers = torch.floor(torch.log2(torch.abs(dlt_search_l))).int()

            positive_vulnerability_powers = torch.zeros_like(dlt_search_h, device=self.device)
            positive_vulnerability_powers = torch.floor(torch.log2(dlt_search_h)).int()
            positive_vulnerability_powers[positive_vulnerability_powers > 10] = 10

            lower_bound_criticality = errors_dist_weight_channel[10 - negative_vulnerability_powers]
            upper_bound_criticality = errors_dist_weight_channel[22 + 10 + positive_vulnerability_powers]
            noncriticality_channel = torch.sum(upper_bound_criticality - lower_bound_criticality)
            
            non_crit_channels[channel] = noncriticality_channel / batch_size


            #free memory
            del channel_grad 
            del grad_bool_map
            #del corrupted_out
            torch.cuda.empty_cache()


        del images
        del x_pad
        torch.cuda.empty_cache()
        return 1 - non_crit_channels
    