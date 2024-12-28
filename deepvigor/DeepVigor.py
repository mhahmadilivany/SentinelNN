import torch
import torch.nn as nn
import torch.nn.functional as F
import net_utils
import math


class DeepVigor():
    def __init__(self, nn_module, network_name, device):
        self.nn_module = nn_module
        self.network_name = network_name
        self.device = device
        
    def delta_injection_neuron(self, neuron, value):
        def hook(model, input, output):
            output_2d = torch.reshape(output, (output.size(0), torch.numel(output[0])))
            output_2d[:, neuron] += value
            output.retain_grad()
            self.activation = output
        return hook

    def delta_injection_channel(self, channel, value):
        def hook(model, input, output):
            output[:, channel] += value.unsqueeze(1).unsqueeze(2).expand(output.size(0), output.size(2), output.size(3))
            output.retain_grad()
            self.activation = output
        return hook
    
    def all_activations_vulnerability_factor(self, images, layer, layer_info_set, out_no, file_name):
        features_count = layer_info_set["neurons_in_layer"]
        stride = layer_info_set["stride"]
        kernel_size = layer_info_set["kernel_size"]
        x_pad = layer_info_set["layer_inputs"]
        cnv_weights = layer_info_set["layer_weights"]
        batch_size = layer_info_set["batch_size"]
        out_channel_count = layer_info_set["out_channel"]
        resolution = 10
        inf_represent = 2 ** resolution
        
        analysis_counters_activations = torch.zeros(12, device=self.device)     #total_neurons, 
                                                                                # analyzed_negative, single_big_negative, multi_big_negative, mixed_negative, skipped_negative, 
                                                                                # analyzed_positive, single_big_positive, multi_big_positive, mixed_positive, skipped_positive

        neurons_in_channel = features_count // out_channel_count
        fmap_width = int(math.sqrt(neurons_in_channel))

        self.activation = torch.tensor([])
        non_crit_neurons = torch.zeros(features_count, device=self.device)
        last_layer_out = self.nn_module.forward(images)
        _, detected_labels = torch.max(last_layer_out, 1)
        analysis_counters_activations[11] += 1
        one_hots = torch.unsqueeze(F.one_hot(detected_labels, num_classes=out_no), 1)
        prev_neuron = 0
        analysis_counters_activations[0] = features_count       #total_neurons

        for channel in range(out_channel_count):
            for neuron_ind in range(neurons_in_channel):
                neuron = channel * neurons_in_channel + neuron_ind
                
                neuron_weights = cnv_weights[channel].unsqueeze(0)
                output_ind_row = neuron_ind // fmap_width
                output_ind_col = neuron_ind  % fmap_width
                input_ind_row = output_ind_row * stride
                input_ind_col = output_ind_col * stride
                sliced_inputs = x_pad[:, :, input_ind_row : input_ind_row + kernel_size, input_ind_col : input_ind_col + kernel_size]
                
                errors_dist_act = net_utils.vulnerability_values_space_act(sliced_inputs, neuron_weights, self.device)
                VVSS_dict_activation = net_utils.creating_VVSS_dict(errors_dist_act, resolution, self.device)
                
                
                #finding deltas in negative numbers
                dlt_search_l = -torch.ones(batch_size, device=self.device)
                self.activation = torch.tensor([])
                handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_l))
                corrupted_out = self.nn_module(images)
                _, corrupted_labels = torch.max(corrupted_out, 1)
                analysis_counters_activations[11] += 1
                handle.remove()
                
                loss_deepvigor = (torch.sum(torch.sigmoid(torch.unsqueeze(torch.sum(corrupted_out * one_hots, 1), 1) - corrupted_out))) / batch_size
                loss_deepvigor.backward()

                neuron_grad = torch.reshape(self.activation.grad.data, (batch_size, features_count))
                neuron_grad[neuron_grad != 0] = 1
                grad_bool_map = torch.eq(neuron_grad[:, neuron], torch.zeros_like(neuron_grad[:, neuron], device=self.device))                        
                
                if torch.sum(neuron_grad, 0)[neuron] != 0:          #there is some images misclassified by faults
                    true_classified = torch.eq(corrupted_labels, detected_labels)
                    #counter for analyzed neurons
                    analysis_counters_activations[1] += 1               #analyzed_negative
                    if torch.sum(true_classified) == batch_size:
                        if VVSS_dict_activation['neg_inf'].size(0) <= 1:          #all images are misclassified by vulnerability_values < -1
                            #counter for single vul value in [-inf, -1]
                            analysis_counters_activations[2] += 1       #single_big_negative
                            dlt_search_l = torch.ones_like(dlt_search_l, device=self.device) * (-inf_represent) #VVSS_dict_activation['neg_inf'][0]
                        else:
                            #counter for multiple vul value in [-inf, -1]
                            analysis_counters_activations[3] += 1       #multi_big_negative
                            vector_len = VVSS_dict_activation['neg_inf'].size(0)
                            iteration_count = vector_len // 2
                            index_tensor = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count
                            for _ in range(iteration_count, 0, -1):
                                dlt_search_l = VVSS_dict_activation['neg_inf'][index_tensor]
                                handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_l))
                                corrupted_out = self.nn_module(images)
                                _, corrupted_labels = torch.max(corrupted_out, 1)
                                true_classified = torch.eq(corrupted_labels, detected_labels)
                                analysis_counters_activations[11] += 1
                                index_tensor = torch.logical_not(true_classified) * (index_tensor + 1) + true_classified * (index_tensor - 1)
                                index_tensor[index_tensor >= vector_len] = vector_len - 1
                                index_tensor[index_tensor < 0] = 0
                                handle.remove()
                                
                            dlt_search_l = VVSS_dict_activation['neg_inf'][index_tensor]
                            dlt_search_l[grad_bool_map == 1] = VVSS_dict_activation['neg_inf'][0]
                    
                    else:       #images misclassified by vulnerability_values < 0
                        #counter for -inf < vul < 0
                        analysis_counters_activations[4] += 1       #mixed_negative
                        true_classified_init = torch.clone(torch.logical_not(true_classified))
                        vector_len_inf = VVSS_dict_activation['neg_inf'].size(0)
                        iteration_count_inf = vector_len_inf // 2
                        index_tensor_inf = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_inf
                        vector_len_1 = VVSS_dict_activation['neg_1'].size(0)
                        iteration_count_1 = vector_len_1 // 2     #will be bigger value
                        index_tensor_1 = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_1
                        
                        for _ in range(iteration_count_1, 0, -1):
                            dlt_search_l = torch.logical_not(true_classified_init) * VVSS_dict_activation['neg_1'][index_tensor_1] + true_classified_init * VVSS_dict_activation['neg_inf'][index_tensor_inf]
                            handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_l))
                            corrupted_out = self.nn_module(images)
                            _, corrupted_labels = torch.max(corrupted_out, 1)
                            true_classified = torch.eq(corrupted_labels, detected_labels)
                            analysis_counters_activations[11] += 1
                            
                            index_tensor_inf = true_classified_init * (torch.logical_not(true_classified) * (index_tensor_inf + 1) + true_classified * (index_tensor_inf - 1))
                            index_tensor_1 = torch.logical_not(true_classified_init) * (torch.logical_not(true_classified) * (index_tensor_1 + 1) + true_classified * (index_tensor_1 - 1))
                            index_tensor_inf[index_tensor_inf >= vector_len_inf] = vector_len_inf - 1
                            index_tensor_inf[index_tensor_inf < 0] = 0
                            index_tensor_1[index_tensor_1 >= vector_len_1] = vector_len_1 - 1
                            index_tensor_1[index_tensor_1 < 0] = 0
                            handle.remove()
                        dlt_search_l = torch.logical_not(true_classified_init) * VVSS_dict_activation['neg_1'][index_tensor_1] + true_classified_init * VVSS_dict_activation['neg_inf'][index_tensor_inf]
                        dlt_search_l[grad_bool_map == 1] = VVSS_dict_activation['neg_inf'][0]

                else:
                    #counter for vul < -inf
                    analysis_counters_activations[5] += 1   #skipped_negative
                    dlt_search_l = torch.ones_like(dlt_search_l, device=self.device) * VVSS_dict_activation['neg_inf'][0]  #-inf_represent
                

                #free memory
                del self.activation.grad
                del loss_deepvigor
                del neuron_grad 
                del grad_bool_map
                del corrupted_out
                torch.cuda.empty_cache()
                
                #finding deltas in positive numbers
                dlt_search_h = torch.ones(batch_size, device=self.device)
                self.activation = torch.tensor([])
                handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_h))
                corrupted_out = self.nn_module(images)
                _, corrupted_labels = torch.max(corrupted_out, 1)
                analysis_counters_activations[11] += 1
                handle.remove()

                loss_deepvigor = (torch.sum(torch.sigmoid(torch.unsqueeze(torch.sum(corrupted_out * one_hots, 1), 1) - corrupted_out))) / batch_size
                loss_deepvigor.backward()

                neuron_grad = torch.reshape(self.activation.grad.data, (batch_size, features_count))
                neuron_grad[neuron_grad != 0] = 1
                grad_bool_map = torch.eq(neuron_grad[:, neuron], torch.zeros_like(neuron_grad[:, neuron], device=self.device))

                if torch.sum(neuron_grad, 0)[neuron] != 0:          #there is some images misclassified by faults
                    true_classified = torch.eq(corrupted_labels, detected_labels)
                    analysis_counters_activations[6] += 1           #analyzed_positive
                    if torch.sum(true_classified) == batch_size:        #all images are misclassified by vulnerability_values > 1
                        if VVSS_dict_activation['pos_inf'].size(0) == 1:
                            analysis_counters_activations[7] += 1           #single_big_positive
                            dlt_search_h = torch.ones_like(dlt_search_h, device=self.device) * inf_represent
                        else:
                            analysis_counters_activations[8] += 1           #multi_big_positive
                            vector_len = VVSS_dict_activation['pos_inf'].size(0)
                            iteration_count = vector_len // 2
                            index_tensor = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count
                            for _ in range(iteration_count, 0, -1):
                                dlt_search_h = VVSS_dict_activation['pos_inf'][index_tensor]
                                handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_h))
                                corrupted_out = self.nn_module(images)
                                _, corrupted_labels = torch.max(corrupted_out, 1)
                                true_classified = torch.eq(corrupted_labels, detected_labels)
                                index_tensor = torch.logical_not(true_classified) * (index_tensor - 1) + true_classified * (index_tensor + 1)
                                analysis_counters_activations[11] += 1
                                index_tensor[index_tensor >= vector_len] = vector_len - 1
                                index_tensor[index_tensor < 0] = 0
                                handle.remove()
                            dlt_search_h = VVSS_dict_activation['pos_inf'][index_tensor]
                            dlt_search_h[grad_bool_map == 1] = inf_represent     #inf_represent


                    else:
                        analysis_counters_activations[9] += 1           #mixed_positive
                        vector_len_inf = VVSS_dict_activation['pos_inf'].size(0)
                        vector_len_1 = VVSS_dict_activation['pos_1'].size(0)
                        iteration_count_inf = vector_len_inf // 2
                        iteration_count_1 = vector_len_1 // 2           #will be bigger value
                        index_tensor_inf = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_inf
                        index_tensor_1 = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_1
                        true_classified_init = torch.clone(true_classified)
                        for _ in range(iteration_count_1, 0, -1):
                            dlt_search_h = torch.logical_not(true_classified_init) * VVSS_dict_activation['pos_1'][index_tensor_1] + true_classified_init * VVSS_dict_activation['pos_inf'][index_tensor_inf]
                            handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_h))
                            corrupted_out = self.nn_module(images)
                            _, corrupted_labels = torch.max(corrupted_out, 1)
                            true_classified = torch.eq(corrupted_labels, detected_labels)
                            analysis_counters_activations[11] += 1
                            index_tensor_inf = true_classified_init * (torch.logical_not(true_classified) * (index_tensor_inf - 1) + true_classified * (index_tensor_inf + 1))
                            index_tensor_1 = torch.logical_not(true_classified_init) * (torch.logical_not(true_classified) * (index_tensor_1 - 1) + true_classified * (index_tensor_1 + 1))
                            index_tensor_inf[index_tensor_inf >= vector_len_inf] = vector_len_inf - 1
                            index_tensor_inf[index_tensor_inf < 0] = 0
                            index_tensor_1[index_tensor_1 >= vector_len_1] = vector_len_1 - 1
                            index_tensor_1[index_tensor_1 < 0] = 0
                            handle.remove()
                        dlt_search_h = torch.logical_not(true_classified_init) * VVSS_dict_activation['pos_1'][index_tensor_1] + true_classified_init * VVSS_dict_activation['pos_inf'][index_tensor_inf]
                        dlt_search_h[grad_bool_map == 1] = inf_represent
                else:
                    analysis_counters_activations[10] += 1           #skipped_positive
                    dlt_search_h = torch.ones_like(dlt_search_h, device=self.device) * inf_represent
                
                #free memory
                del self.activation.grad
                del loss_deepvigor
                del neuron_grad
                del grad_bool_map
                del corrupted_out
                self.activation = torch.tensor([])
                torch.cuda.empty_cache()

                #obtaining the error distribution for the neuron's outputs            
                #NVF calculation
                negative_vulnerability_powers = torch.zeros_like(dlt_search_l, device=self.device)
                negative_vulnerability_powers = torch.floor(torch.log2(torch.abs(dlt_search_l))).int()

                positive_vulnerability_powers = torch.zeros_like(dlt_search_h, device=self.device)
                positive_vulnerability_powers = torch.floor(torch.log2(dlt_search_h)).int()
                positive_vulnerability_powers[positive_vulnerability_powers > resolution] = resolution

                lower_bound_criticality = errors_dist_act[resolution - negative_vulnerability_powers]
                upper_bound_criticality = errors_dist_act[22 + 10 + positive_vulnerability_powers]
                noncriticality_neuron = torch.sum(upper_bound_criticality - lower_bound_criticality)
                
                non_crit_neurons[neuron] = noncriticality_neuron / batch_size

                del neuron_weights
                del sliced_inputs
                del errors_dist_act
                del negative_vulnerability_powers
                del positive_vulnerability_powers
                del dlt_search_l
                del dlt_search_h
                torch.cuda.empty_cache()

                #temporary saving results
                if neuron % 4096 == 0:
                    noncritical_nvf_file = open(file_name, 'a')
                    for tmp_neuron in range(prev_neuron, neuron):
                        noncritical_nvf_file.write(str(non_crit_neurons[tmp_neuron].item()) + "\n")
                    noncritical_nvf_file.close()
                    prev_neuron = neuron

        torch.cuda.empty_cache()
        return non_crit_neurons, analysis_counters_activations

    def all_weights_vulnerability_factor(self, images, layer, layer_info_set, out_no):
        features_count = layer_info_set["neurons_in_layer"]
        stride = layer_info_set["stride"]
        kernel_size = layer_info_set["kernel_size"]
        x_pad = layer_info_set["layer_inputs"]
        cnv_weights = layer_info_set["layer_weights"]
        batch_size = layer_info_set["batch_size"]
        out_channel_count = layer_info_set["out_channel"]
        resolution = 10
        inf_represent = 2 ** resolution
        analysis_counters_weights = torch.zeros(12, device=self.device)     #total_neurons, 
                                                                            # analyzed_negative, single_big_negative, multi_big_negative, mixed_negative, skipped_negative, 
                                                                            # analyzed_positive, single_big_positive, multi_big_positive, mixed_positive, skipped_positive, total_forward_count

        neurons_in_channel = features_count // out_channel_count
        fmap_width = int(math.sqrt(neurons_in_channel))

        self.activation = torch.tensor([])
        non_crit_channels = torch.zeros(out_channel_count, device=self.device)
        last_layer_out = self.nn_module.forward(images)
        analysis_counters_weights[11] += 1
        _, detected_labels = torch.max(last_layer_out, 1)
        one_hots = torch.unsqueeze(F.one_hot(detected_labels, num_classes=out_no), 1)
        analysis_counters_weights[0] = out_channel_count

        for channel in range(out_channel_count):
            errors_dist_weight_channel = torch.zeros(4 * resolution + 3, device=self.device)
            for neuron_ind in range(neurons_in_channel):
                neuron_weights = cnv_weights[channel].unsqueeze(0)
                output_ind_row = neuron_ind // fmap_width
                output_ind_col = neuron_ind  % fmap_width
                input_ind_row = output_ind_row * stride
                input_ind_col = output_ind_col * stride
                sliced_inputs = x_pad[:, :, input_ind_row : input_ind_row + kernel_size, input_ind_col : input_ind_col + kernel_size]
                
                errors_dist_weight_neuron = net_utils.vulnerability_values_space_weight(sliced_inputs, neuron_weights, self.device)
                errors_dist_weight_channel += errors_dist_weight_neuron
 
            #analysis for faulty weights 
            errors_dist_weight_channel = errors_dist_weight_channel / neurons_in_channel
            VVSS_dict_weights = net_utils.creating_VVSS_dict(errors_dist_weight_channel, resolution, self.device)
            
            #finding deltas in negative numbers
            dlt_search_l = -torch.ones(batch_size, device=self.device)
            self.activation = torch.tensor([])
            handle = layer.register_forward_hook(self.delta_injection_channel(channel, dlt_search_l))
            corrupted_out = self.nn_module(images)
            _, corrupted_labels = torch.max(corrupted_out, 1)
            analysis_counters_weights[11] += 1
            handle.remove()
            
            loss_deepvigor = (torch.sum(torch.sigmoid(torch.unsqueeze(torch.sum(corrupted_out * one_hots, 1), 1) - corrupted_out))) / batch_size
            loss_deepvigor.backward()

            channel_grad = torch.sum(torch.sum(self.activation.grad.data, 3), 2)
            channel_grad[channel_grad != 0] = 1
            grad_bool_map = torch.eq(channel_grad[:, channel], torch.zeros_like(channel_grad[:, channel], device=self.device))

            if torch.sum(channel_grad, 0)[channel] != 0:           #there is some images misclassified by faults
                true_classified = torch.eq(corrupted_labels, detected_labels)
                analysis_counters_weights[1] += 1               #analyzed_negative
                if torch.sum(true_classified) == batch_size:
                    if VVSS_dict_weights['neg_inf'].size(0) <= 1:               #all images are misclassified by vulnerability_values < -1
                        analysis_counters_weights[2] += 1       #single_big_negative
                        dlt_search_l = torch.ones_like(dlt_search_l, device=self.device) * (-inf_represent)
                    else:
                        analysis_counters_weights[3] += 1       #multi_big_negative
                        vector_len = VVSS_dict_weights['neg_inf'].size(0)
                        iteration_count = vector_len // 2
                        index_tensor = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count
                        for _ in range(iteration_count, 0, -1):
                            dlt_search_l = VVSS_dict_weights['neg_inf'][index_tensor]
                            handle = layer.register_forward_hook(self.delta_injection_channel(channel, dlt_search_l))
                            corrupted_out = self.nn_module(images)
                            _, corrupted_labels = torch.max(corrupted_out, 1)
                            analysis_counters_weights[11] += 1
                            true_classified = torch.eq(corrupted_labels, detected_labels)
                            index_tensor = torch.logical_not(true_classified) * (index_tensor + 1) + true_classified * (index_tensor - 1)
                            index_tensor[index_tensor >= vector_len] = vector_len - 1
                            index_tensor[index_tensor < 0] = 0
                            handle.remove()
                            
                        dlt_search_l = VVSS_dict_weights['neg_inf'][index_tensor]
                        dlt_search_l[grad_bool_map == 1] = -inf_represent
                
                else:       #images misclassified by vulnerability_values < 0
                    analysis_counters_weights[4] += 1       #mixed_negative
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
                        corrupted_out = self.nn_module(images)
                        _, corrupted_labels = torch.max(corrupted_out, 1)
                        true_classified = torch.eq(corrupted_labels, detected_labels)
                        analysis_counters_weights[11] += 1
                        
                        index_tensor_inf = true_classified_init * (torch.logical_not(true_classified) * (index_tensor_inf + 1) + true_classified * (index_tensor_inf - 1))
                        index_tensor_1 = torch.logical_not(true_classified_init) * (torch.logical_not(true_classified) * (index_tensor_1 + 1) + true_classified * (index_tensor_1 - 1))
                        index_tensor_inf[index_tensor_inf >= vector_len_inf] = vector_len_inf - 1
                        index_tensor_inf[index_tensor_inf < 0] = 0
                        index_tensor_1[index_tensor_1 >= vector_len_1] = vector_len_1 - 1
                        index_tensor_1[index_tensor_1 < 0] = 0
                        handle.remove()
                    dlt_search_l = torch.logical_not(true_classified_init) * VVSS_dict_weights['neg_1'][index_tensor_1] + true_classified_init * VVSS_dict_weights['neg_inf'][index_tensor_inf]
                    dlt_search_l[grad_bool_map == 1] = -inf_represent

            else:
                #counter for vul < -inf
                analysis_counters_weights[5] += 1   #skipped_negative
                dlt_search_l = torch.ones_like(dlt_search_l, device=self.device) * -inf_represent 

            #free memory
            del self.activation.grad
            del loss_deepvigor
            del channel_grad 
            del grad_bool_map
            del corrupted_out
            torch.cuda.empty_cache()

            #finding deltas in positive numbers
            dlt_search_h = torch.ones(batch_size, device=self.device)
            self.activation = torch.tensor([])
            handle = layer.register_forward_hook(self.delta_injection_channel(channel, dlt_search_h))
            corrupted_out = self.nn_module(images)
            _, corrupted_labels = torch.max(corrupted_out, 1)
            analysis_counters_weights[11] += 1
            handle.remove()

            loss_deepvigor = (torch.sum(torch.sigmoid(torch.unsqueeze(torch.sum(corrupted_out * one_hots, 1), 1) - corrupted_out))) / batch_size
            loss_deepvigor.backward()

            channel_grad = torch.sum(torch.sum(self.activation.grad.data, 3), 2) 
            channel_grad[channel_grad != 0] = 1
            grad_bool_map = torch.eq(channel_grad[:, channel], torch.zeros_like(channel_grad[:, channel], device=self.device))

            if torch.sum(channel_grad, 0)[channel] != 0:          #there is some images misclassified by faults
                true_classified = torch.eq(corrupted_labels, detected_labels)
                analysis_counters_weights[6] += 1           #analyzed_positive
                if torch.sum(true_classified) == batch_size:        #all images are misclassified by vulnerability_values > 1
                    if VVSS_dict_weights['pos_inf'].size(0) <= 1:
                        analysis_counters_weights[7] += 1           #single_big_positive
                        dlt_search_h = torch.ones_like(dlt_search_h, device=self.device) * inf_represent 
                    else:
                        analysis_counters_weights[8] += 1           #multi_big_positive
                        vector_len = VVSS_dict_weights['pos_inf'].size(0)
                        iteration_count = vector_len // 2
                        index_tensor = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count
                        for _ in range(iteration_count, 0, -1):
                            dlt_search_h = VVSS_dict_weights['pos_inf'][index_tensor]
                            handle = layer.register_forward_hook(self.delta_injection_channel(channel, dlt_search_h))
                            corrupted_out = self.nn_module(images)
                            _, corrupted_labels = torch.max(corrupted_out, 1)
                            true_classified = torch.eq(corrupted_labels, detected_labels)
                            analysis_counters_weights[11] += 1
                            index_tensor = torch.logical_not(true_classified) * (index_tensor - 1) + true_classified * (index_tensor + 1)
                            index_tensor[index_tensor >= vector_len] = vector_len - 1
                            index_tensor[index_tensor < 0] = 0
                            handle.remove()
                        dlt_search_h = VVSS_dict_weights['pos_inf'][index_tensor]
                        dlt_search_h[grad_bool_map == 1] = inf_represent  

                else:
                    analysis_counters_weights[9] += 1           #mixed_positive
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
                        corrupted_out = self.nn_module(images)
                        _, corrupted_labels = torch.max(corrupted_out, 1)
                        true_classified = torch.eq(corrupted_labels, detected_labels)
                        analysis_counters_weights[11] += 1
                        index_tensor_inf = true_classified_init * (torch.logical_not(true_classified) * (index_tensor_inf - 1) + true_classified * (index_tensor_inf + 1))
                        index_tensor_1 = torch.logical_not(true_classified_init) * (torch.logical_not(true_classified) * (index_tensor_1 - 1) + true_classified * (index_tensor_1 + 1))
                        index_tensor_inf[index_tensor_inf >= vector_len_inf] = vector_len_inf - 1
                        index_tensor_inf[index_tensor_inf < 0] = 0
                        index_tensor_1[index_tensor_1 >= vector_len_1] = vector_len_1 - 1
                        index_tensor_1[index_tensor_1 < 0] = 0
                        handle.remove()
                    dlt_search_h = torch.logical_not(true_classified_init) * VVSS_dict_weights['pos_1'][index_tensor_1] + true_classified_init * VVSS_dict_weights['pos_inf'][index_tensor_inf]
                    dlt_search_h[grad_bool_map == 1] = inf_represent  
            else:
                analysis_counters_weights[10] += 1           #skipped_positive
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
            del self.activation.grad
            del loss_deepvigor
            del channel_grad 
            del grad_bool_map
            del corrupted_out
            torch.cuda.empty_cache()


        torch.cuda.empty_cache()
        return non_crit_channels, analysis_counters_weights
    
    def sampling_activations_vulnerability_factor(self, images, layer, layer_info_set, ch_sampling_ratio, out_no):
        features_count = layer_info_set["neurons_in_layer"]
        stride = layer_info_set["stride"]
        kernel_size = layer_info_set["kernel_size"]
        x_pad = layer_info_set["layer_inputs"]
        cnv_weights = layer_info_set["layer_weights"]
        batch_size = layer_info_set["batch_size"]
        out_channel_count = layer_info_set["out_channel"]
        resolution = 10
        inf_represent = 2 ** resolution
        
        analysis_counters_activations = torch.zeros(12, device=self.device)     #analyzed_neurons, 
                                                                                # analyzed_negative, single_big_negative, multi_big_negative, mixed_negative, skipped_negative, 
                                                                                # analyzed_positive, single_big_positive, multi_big_positive, mixed_positive, skipped_positive


        out_channel_samples = torch.max(torch.tensor([int(ch_sampling_ratio * out_channel_count), 1])).item()
        neurons_in_channel = features_count // out_channel_count
        fmap_width = int(math.sqrt(neurons_in_channel))

        self.activation = torch.tensor([])
        non_crit_neurons = torch.zeros(features_count, device=self.device)
        last_layer_out = self.nn_module.forward(images)
        _, detected_labels = torch.max(last_layer_out, 1)
        one_hots = torch.unsqueeze(F.one_hot(detected_labels, num_classes=out_no), 1)

        neurons_samples = torch.max(torch.tensor([int(torch.log2(torch.tensor([neurons_in_channel]))), 1])).item()
        channels_set = torch.tensor([])
        
        sampled_CVF = torch.zeros(out_channel_count, device=self.device)
        ch_counter = 0
        while ch_counter < out_channel_samples:
            rand_channel = torch.randint(out_channel_count, (1,)).item()
            if rand_channel not in channels_set:
                ch_counter += 1
                channels_set = torch.cat((channels_set, torch.tensor([rand_channel])))
                neurons_set = torch.tensor([])
                nrn_counter = 0
                NVF_ch = 0
                
                while nrn_counter < neurons_samples:
                    rand_neuron = torch.randint(neurons_in_channel, (1,)).item()
                    if rand_neuron not in neurons_set:
                        nrn_counter += 1
                        neurons_set = torch.cat((neurons_set, torch.tensor([rand_neuron])))
                        analysis_counters_activations[0] += 1

                        neuron = rand_channel * neurons_in_channel + rand_neuron

                        # errors analysis based the faulty weights and activations 
                        #                           to obtain vulnerability values
                        neuron_weights = cnv_weights[rand_channel].unsqueeze(0)

                        output_ind_row = rand_neuron // fmap_width
                        output_ind_col = rand_neuron % fmap_width
                        input_ind_row = output_ind_row * stride
                        input_ind_col = output_ind_col * stride
                        sliced_inputs = x_pad[:, :, input_ind_row : input_ind_row + kernel_size, input_ind_col : input_ind_col + kernel_size]

                        errors_dist_act = net_utils.vulnerability_values_space_act(sliced_inputs, neuron_weights, self.device)
                        VVSS_dict_activation = net_utils.creating_VVSS_dict(errors_dist_act, resolution, self.device)

                        #finding deltas in negative numbers
                        dlt_search_l = -torch.ones(batch_size, device=self.device)
                        self.activation = torch.tensor([])
                        handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_l))
                        corrupted_out = self.nn_module(images)
                        _, corrupted_labels = torch.max(corrupted_out, 1)
                        analysis_counters_activations[11] += 1
                        handle.remove()
                        
                        loss_deepvigor = (torch.sum(torch.sigmoid(torch.unsqueeze(torch.sum(corrupted_out * one_hots, 1), 1) - corrupted_out))) / batch_size
                        loss_deepvigor.backward()

                        neuron_grad = torch.reshape(self.activation.grad.data, (batch_size, features_count))
                        neuron_grad[neuron_grad != 0] = 1
                        grad_bool_map = torch.eq(neuron_grad[:, neuron], torch.zeros_like(neuron_grad[:, neuron], device=self.device)) 

                        if torch.sum(neuron_grad, 0)[neuron] != 0:          #there is some images misclassified by faults
                            true_classified = torch.eq(corrupted_labels, detected_labels)
                            analysis_counters_activations[1] += 1               #analyzed_negative
                            if torch.sum(true_classified) == batch_size:
                                if VVSS_dict_activation['neg_inf'].size(0) <= 1:          #all images are misclassified by vulnerability_values < -1
                                    #counter for single vul value in [-inf, -1]
                                    analysis_counters_activations[2] += 1       #single_big_negative
                                    dlt_search_l = torch.ones_like(dlt_search_l, device=self.device) * (-inf_represent)
                                else:
                                    #counter for multiple vul value in [-inf, -1]
                                    analysis_counters_activations[3] += 1       #multi_big_negative
                                    vector_len = VVSS_dict_activation['neg_inf'].size(0)
                                    iteration_count = vector_len // 2
                                    index_tensor = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count
                                    for _ in range(iteration_count, 0, -1):
                                        dlt_search_l = VVSS_dict_activation['neg_inf'][index_tensor]
                                        handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_l))
                                        corrupted_out = self.nn_module(images)
                                        _, corrupted_labels = torch.max(corrupted_out, 1)
                                        true_classified = torch.eq(corrupted_labels, detected_labels)
                                        analysis_counters_activations[11] += 1
                                        index_tensor = torch.logical_not(true_classified) * (index_tensor + 1) + true_classified * (index_tensor - 1)
                                        index_tensor[index_tensor >= vector_len] = vector_len - 1
                                        index_tensor[index_tensor < 0] = 0
                                        handle.remove()
                                        
                                    dlt_search_l = VVSS_dict_activation['neg_inf'][index_tensor]
                                    dlt_search_l[grad_bool_map == 1] = VVSS_dict_activation['neg_inf'][0]

                            else:       #images misclassified by vulnerability_values < 0
                                analysis_counters_activations[4] += 1       #mixed_negative
                                true_classified_init = torch.clone(torch.logical_not(true_classified))
                                vector_len_inf = VVSS_dict_activation['neg_inf'].size(0)
                                iteration_count_inf = vector_len_inf // 2
                                index_tensor_inf = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_inf
                                vector_len_1 = VVSS_dict_activation['neg_1'].size(0)
                                iteration_count_1 = vector_len_1 // 2     #will be bigger value
                                index_tensor_1 = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_1
                                
                                for _ in range(iteration_count_1, 0, -1):
                                    dlt_search_l = torch.logical_not(true_classified_init) * VVSS_dict_activation['neg_1'][index_tensor_1] + true_classified_init * VVSS_dict_activation['neg_inf'][index_tensor_inf]
                                    handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_l))
                                    corrupted_out = self.nn_module(images)
                                    _, corrupted_labels = torch.max(corrupted_out, 1)
                                    true_classified = torch.eq(corrupted_labels, detected_labels)
                                    analysis_counters_activations[11] += 1
                                    
                                    index_tensor_inf = true_classified_init * (torch.logical_not(true_classified) * (index_tensor_inf + 1) + true_classified * (index_tensor_inf - 1))
                                    index_tensor_1 = torch.logical_not(true_classified_init) * (torch.logical_not(true_classified) * (index_tensor_1 + 1) + true_classified * (index_tensor_1 - 1))
                                    index_tensor_inf[index_tensor_inf >= vector_len_inf] = vector_len_inf - 1
                                    index_tensor_inf[index_tensor_inf < 0] = 0
                                    index_tensor_1[index_tensor_1 >= vector_len_1] = vector_len_1 - 1
                                    index_tensor_1[index_tensor_1 < 0] = 0
                                    handle.remove()
                                dlt_search_l = torch.logical_not(true_classified_init) * VVSS_dict_activation['neg_1'][index_tensor_1] + true_classified_init * VVSS_dict_activation['neg_inf'][index_tensor_inf]
                                dlt_search_l[grad_bool_map == 1] = -inf_represent
                        else:
                            #counter for vul < -inf
                            analysis_counters_activations[5] += 1   #skipped_negative
                            dlt_search_l = torch.ones_like(dlt_search_l, device=self.device) * -inf_represent
                        
                        #free memory
                        del self.activation.grad
                        del loss_deepvigor
                        del neuron_grad
                        del grad_bool_map
                        del corrupted_out
                        torch.cuda.empty_cache()
                        
                        #finding deltas in positive numbers
                        dlt_search_h = torch.ones(batch_size, device=self.device)
                        self.activation = torch.tensor([])
                        handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_h))
                        corrupted_out = self.nn_module(images)
                        _, corrupted_labels = torch.max(corrupted_out, 1)
                        analysis_counters_activations[11] += 1
                        handle.remove()

                        loss_deepvigor = (torch.sum(torch.sigmoid(torch.unsqueeze(torch.sum(corrupted_out * one_hots, 1), 1) - corrupted_out))) / batch_size
                        loss_deepvigor.backward()

                        neuron_grad = torch.reshape(self.activation.grad.data, (batch_size, features_count))
                        neuron_grad[neuron_grad != 0] = 1
                        grad_bool_map = torch.eq(neuron_grad[:, neuron], torch.zeros_like(neuron_grad[:, neuron], device=self.device))

                        
                        if torch.sum(neuron_grad, 0)[neuron] != 0:          #there is some images misclassified by faults
                            true_classified = torch.eq(corrupted_labels, detected_labels)
                            analysis_counters_activations[6] += 1           #analyzed_positive
                            if torch.sum(true_classified) == batch_size:        #all images are misclassified by vulnerability_values > 1
                                if VVSS_dict_activation['pos_inf'].size(0) == 1:
                                    analysis_counters_activations[7] += 1           #single_big_positive
                                    dlt_search_h = torch.ones_like(dlt_search_h, device=self.device) * inf_represent
                                else:
                                    analysis_counters_activations[8] += 1           #multi_big_positive
                                    vector_len = VVSS_dict_activation['pos_inf'].size(0)
                                    iteration_count = vector_len // 2
                                    index_tensor = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count
                                    for _ in range(iteration_count, 0, -1):
                                        dlt_search_h = VVSS_dict_activation['pos_inf'][index_tensor]
                                        handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_h))
                                        corrupted_out = self.nn_module(images)
                                        _, corrupted_labels = torch.max(corrupted_out, 1)
                                        true_classified = torch.eq(corrupted_labels, detected_labels)
                                        index_tensor = torch.logical_not(true_classified) * (index_tensor - 1) + true_classified * (index_tensor + 1)
                                        analysis_counters_activations[11] += 1
                                        index_tensor[index_tensor >= vector_len] = vector_len - 1
                                        index_tensor[index_tensor < 0] = 0
                                        handle.remove()
                                    dlt_search_h = VVSS_dict_activation['pos_inf'][index_tensor]
                                    dlt_search_h[grad_bool_map == 1] = inf_represent

                            else:
                                analysis_counters_activations[9] += 1           #mixed_positive
                                vector_len_inf = VVSS_dict_activation['pos_inf'].size(0)
                                vector_len_1 = VVSS_dict_activation['pos_1'].size(0)
                                iteration_count_inf = vector_len_inf // 2
                                iteration_count_1 = vector_len_1 // 2           #will be bigger value
                                index_tensor_inf = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_inf
                                index_tensor_1 = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_1
                                true_classified_init = torch.clone(true_classified)
                                for _ in range(iteration_count_1, 0, -1):
                                    dlt_search_h = torch.logical_not(true_classified_init) * VVSS_dict_activation['pos_1'][index_tensor_1] + true_classified_init * VVSS_dict_activation['pos_inf'][index_tensor_inf]
                                    handle = layer.register_forward_hook(self.delta_injection_neuron(neuron, dlt_search_h))
                                    corrupted_out = self.nn_module(images)
                                    _, corrupted_labels = torch.max(corrupted_out, 1)
                                    true_classified = torch.eq(corrupted_labels, detected_labels)
                                    analysis_counters_activations[11] += 1
                                    index_tensor_inf = true_classified_init * (torch.logical_not(true_classified) * (index_tensor_inf - 1) + true_classified * (index_tensor_inf + 1))
                                    index_tensor_1 = torch.logical_not(true_classified_init) * (torch.logical_not(true_classified) * (index_tensor_1 - 1) + true_classified * (index_tensor_1 + 1))
                                    index_tensor_inf[index_tensor_inf >= vector_len_inf] = vector_len_inf - 1
                                    index_tensor_inf[index_tensor_inf < 0] = 0
                                    index_tensor_1[index_tensor_1 >= vector_len_1] = vector_len_1 - 1
                                    index_tensor_1[index_tensor_1 < 0] = 0
                                    handle.remove()
                                dlt_search_h = torch.logical_not(true_classified_init) * VVSS_dict_activation['pos_1'][index_tensor_1] + true_classified_init * VVSS_dict_activation['pos_inf'][index_tensor_inf]
                                dlt_search_h[grad_bool_map == 1] = inf_represent
                        else:
                            analysis_counters_activations[10] += 1           #skipped_positive
                            dlt_search_h = torch.ones_like(dlt_search_h, device=self.device) * inf_represent
                        
                        #free memory
                        del self.activation.grad
                        del loss_deepvigor
                        del neuron_grad
                        del grad_bool_map
                        del corrupted_out
                        self.activation = torch.tensor([])
                        torch.cuda.empty_cache()

                        
                        #NVF calculation
                        negative_vulnerability_powers = torch.zeros_like(dlt_search_l, device=self.device)
                        negative_vulnerability_powers = torch.floor(torch.log2(torch.abs(dlt_search_l))).int()

                        positive_vulnerability_powers = torch.zeros_like(dlt_search_h, device=self.device)
                        positive_vulnerability_powers = torch.floor(torch.log2(dlt_search_h)).int()
                        positive_vulnerability_powers[positive_vulnerability_powers > resolution] = resolution

                        lower_bound_criticality = errors_dist_act[resolution - negative_vulnerability_powers]
                        upper_bound_criticality = errors_dist_act[22 + 10 + positive_vulnerability_powers]
                        noncriticality_neuron = torch.sum(upper_bound_criticality - lower_bound_criticality)
                        
                        non_crit_neurons[neuron] = noncriticality_neuron / batch_size

                        del neuron_weights
                        del sliced_inputs
                        del negative_vulnerability_powers
                        del positive_vulnerability_powers
                        del dlt_search_l
                        del dlt_search_h
                        torch.cuda.empty_cache()

                        NVF_ch += noncriticality_neuron / batch_size
                
                del neurons_set
                sampled_CVF[rand_channel] = NVF_ch / nrn_counter

        torch.cuda.empty_cache()
        
        return non_crit_neurons, analysis_counters_activations, sampled_CVF
    
    def sampling_weights_vulnerability_factor(self, images, layer, layer_info_set, ch_sampling_ratio, out_no):
        features_count = layer_info_set["neurons_in_layer"]
        stride = layer_info_set["stride"]
        kernel_size = layer_info_set["kernel_size"]
        x_pad = layer_info_set["layer_inputs"]
        cnv_weights = layer_info_set["layer_weights"]
        batch_size = layer_info_set["batch_size"]
        out_channel_count = layer_info_set["out_channel"]
        resolution = 10
        inf_represent = 2 ** resolution
        analysis_counters_weights = torch.zeros(12, device=self.device)     #analyzed_neurons, 
                                                                                # analyzed_negative, single_big_negative, multi_big_negative, mixed_negative, skipped_negative, 
                                                                                # analyzed_positive, single_big_positive, multi_big_positive, mixed_positive, skipped_positive

        out_channel_samples = torch.max(torch.tensor([int(ch_sampling_ratio * out_channel_count), 1])).item()
        
        neurons_in_channel = features_count // out_channel_count
        fmap_width = int(math.sqrt(neurons_in_channel))

        self.activation = torch.tensor([])
        non_crit_channels = torch.zeros(out_channel_count, device=self.device)
        last_layer_out = self.nn_module.forward(images)
        _, detected_labels = torch.max(last_layer_out, 1)
        analysis_counters_weights[11] += 1
        one_hots = torch.unsqueeze(F.one_hot(detected_labels, num_classes=out_no), 1)
        neurons_samples = torch.max(torch.tensor([int(torch.log2(torch.tensor([neurons_in_channel]))), 1])).item()

        channels_set = torch.tensor([])
        ch_counter = 0

        while ch_counter < out_channel_samples:
            rand_channel = torch.randint(out_channel_count, (1,)).item()
            if rand_channel not in channels_set:
                ch_counter += 1
                channels_set = torch.cat((channels_set, torch.tensor([rand_channel])))
                neurons_set = torch.tensor([])
                nrn_counter = 0
                NVF_ch = 0
                
                errors_dist_weight_channel = torch.zeros(4 * resolution + 3, device=self.device)
                while nrn_counter < neurons_samples:
                    rand_neuron = torch.randint(neurons_in_channel, (1,)).item()
                    if rand_neuron not in neurons_set:
                        nrn_counter += 1
                        neurons_set = torch.cat((neurons_set, torch.tensor([rand_neuron])))
                        analysis_counters_weights[0] += 1

                        #neuron = rand_channel * neurons_in_channel + rand_neuron
                        neuron_weights = cnv_weights[rand_channel].unsqueeze(0)
                        output_ind_row = rand_neuron // fmap_width
                        output_ind_col = rand_neuron % fmap_width
                        input_ind_row = output_ind_row * stride
                        input_ind_col = output_ind_col * stride
                        sliced_inputs = x_pad[:, :, input_ind_row : input_ind_row + kernel_size, input_ind_col : input_ind_col + kernel_size]

                        errors_dist_weight_neuron = net_utils.vulnerability_values_space_weight(sliced_inputs, neuron_weights, self.device)
                        errors_dist_weight_channel += errors_dist_weight_neuron

                errors_dist_weight_channel = errors_dist_weight_channel / nrn_counter
                VVSS_dict_weights = net_utils.creating_VVSS_dict(errors_dist_weight_channel, resolution, self.device)
                
                #finding deltas in negative numbers
                dlt_search_l = -torch.ones(batch_size, device=self.device)
                self.activation = torch.tensor([])
                handle = layer.register_forward_hook(self.delta_injection_channel(rand_channel, dlt_search_l))
                corrupted_out = self.nn_module(images)
                _, corrupted_labels = torch.max(corrupted_out, 1)
                analysis_counters_weights[11] += 1
                handle.remove()
                
                loss_deepvigor = (torch.sum(torch.sigmoid(torch.unsqueeze(torch.sum(corrupted_out * one_hots, 1), 1) - corrupted_out))) / batch_size
                loss_deepvigor.backward()

                channel_grad = torch.sum(torch.sum(self.activation.grad.data, 3), 2) #torch.reshape(self.activation.grad.data, (batch_size, out_channel_count, neurons_in_channel))
                channel_grad[channel_grad != 0] = 1
                grad_bool_map = torch.eq(channel_grad[:, rand_channel], torch.zeros_like(channel_grad[:, rand_channel], device=self.device))

                if torch.sum(channel_grad, 0)[rand_channel] != 0:           #there is some images misclassified by faults
                    true_classified = torch.eq(corrupted_labels, detected_labels)
                    analysis_counters_weights[1] += 1               #analyzed_negative
                    if torch.sum(true_classified) == batch_size:
                        if VVSS_dict_weights['neg_inf'].size(0) <= 1:               #all images are misclassified by vulnerability_values < -1
                            analysis_counters_weights[2] += 1       #single_big_negative
                            dlt_search_l = torch.ones_like(dlt_search_l, device=self.device) * (-inf_represent)
                        else:
                            analysis_counters_weights[3] += 1       #multi_big_negative
                            vector_len = VVSS_dict_weights['neg_inf'].size(0)
                            iteration_count = vector_len // 2
                            index_tensor = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count
                            for _ in range(iteration_count, 0, -1):
                                dlt_search_l = VVSS_dict_weights['neg_inf'][index_tensor]
                                handle = layer.register_forward_hook(self.delta_injection_channel(rand_channel, dlt_search_l))
                                corrupted_out = self.nn_module(images)
                                _, corrupted_labels = torch.max(corrupted_out, 1)
                                analysis_counters_weights[11] += 1
                                true_classified = torch.eq(corrupted_labels, detected_labels)
                                index_tensor = torch.logical_not(true_classified) * (index_tensor + 1) + true_classified * (index_tensor - 1)
                                index_tensor[index_tensor >= vector_len] = vector_len - 1
                                index_tensor[index_tensor < 0] = 0
                                handle.remove()
                                
                            dlt_search_l = VVSS_dict_weights['neg_inf'][index_tensor]
                            dlt_search_l[grad_bool_map == 1] = -inf_represent
                    
                    else:       #images misclassified by vulnerability_values < 0
                        analysis_counters_weights[4] += 1       #mixed_negative
                        true_classified_init = torch.clone(torch.logical_not(true_classified))
                        vector_len_inf = VVSS_dict_weights['neg_inf'].size(0)
                        iteration_count_inf = vector_len_inf // 2
                        index_tensor_inf = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_inf
                        vector_len_1 = VVSS_dict_weights['neg_1'].size(0)
                        iteration_count_1 = vector_len_1 // 2     #will be bigger value
                        index_tensor_1 = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_1
                        
                        for _ in range(iteration_count_1, 0, -1):
                            dlt_search_l = torch.logical_not(true_classified_init) * VVSS_dict_weights['neg_1'][index_tensor_1] + true_classified_init * VVSS_dict_weights['neg_inf'][index_tensor_inf]
                            handle = layer.register_forward_hook(self.delta_injection_channel(rand_channel, dlt_search_l))
                            corrupted_out = self.nn_module(images)
                            _, corrupted_labels = torch.max(corrupted_out, 1)
                            true_classified = torch.eq(corrupted_labels, detected_labels)
                            analysis_counters_weights[11] += 1
                            
                            index_tensor_inf = true_classified_init * (torch.logical_not(true_classified) * (index_tensor_inf + 1) + true_classified * (index_tensor_inf - 1))
                            index_tensor_1 = torch.logical_not(true_classified_init) * (torch.logical_not(true_classified) * (index_tensor_1 + 1) + true_classified * (index_tensor_1 - 1))
                            index_tensor_inf[index_tensor_inf >= vector_len_inf] = vector_len_inf - 1
                            index_tensor_inf[index_tensor_inf < 0] = 0
                            index_tensor_1[index_tensor_1 >= vector_len_1] = vector_len_1 - 1
                            index_tensor_1[index_tensor_1 < 0] = 0
                            handle.remove()
                        dlt_search_l = torch.logical_not(true_classified_init) * VVSS_dict_weights['neg_1'][index_tensor_1] + true_classified_init * VVSS_dict_weights['neg_inf'][index_tensor_inf]
                        dlt_search_l[grad_bool_map == 1] = -inf_represent

                else:
                    #counter for vul < -inf
                    analysis_counters_weights[5] += 1   #skipped_negative
                    dlt_search_l = torch.ones_like(dlt_search_l, device=self.device) * -inf_represent 

                #free memory
                del self.activation.grad
                del loss_deepvigor
                del channel_grad 
                del grad_bool_map
                del corrupted_out
                torch.cuda.empty_cache()

                #finding deltas in positive numbers
                dlt_search_h = torch.ones(batch_size, device=self.device)
                self.activation = torch.tensor([])
                handle = layer.register_forward_hook(self.delta_injection_channel(rand_channel, dlt_search_h))
                corrupted_out = self.nn_module(images)
                _, corrupted_labels = torch.max(corrupted_out, 1)
                analysis_counters_weights[11] += 1
                handle.remove()

                loss_deepvigor = (torch.sum(torch.sigmoid(torch.unsqueeze(torch.sum(corrupted_out * one_hots, 1), 1) - corrupted_out))) / batch_size
                loss_deepvigor.backward()

                channel_grad = torch.sum(torch.sum(self.activation.grad.data, 3), 2) 
                channel_grad[channel_grad != 0] = 1
                grad_bool_map = torch.eq(channel_grad[:, rand_channel], torch.zeros_like(channel_grad[:, rand_channel], device=self.device))

                if torch.sum(channel_grad, 0)[rand_channel] != 0:          #there is some images misclassified by faults
                    true_classified = torch.eq(corrupted_labels, detected_labels)
                    analysis_counters_weights[6] += 1           #analyzed_positive
                    if torch.sum(true_classified) == batch_size:        #all images are misclassified by vulnerability_values > 1
                        if VVSS_dict_weights['pos_inf'].size(0) <= 1:
                            analysis_counters_weights[7] += 1           #single_big_positive
                            dlt_search_h = torch.ones_like(dlt_search_h, device=self.device) * inf_represent 
                        else:
                            analysis_counters_weights[8] += 1           #multi_big_positive
                            vector_len = VVSS_dict_weights['pos_inf'].size(0)
                            iteration_count = vector_len // 2
                            index_tensor = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count
                            for _ in range(iteration_count, 0, -1):
                                dlt_search_h = VVSS_dict_weights['pos_inf'][index_tensor]
                                handle = layer.register_forward_hook(self.delta_injection_channel(rand_channel, dlt_search_h))
                                corrupted_out = self.nn_module(images)
                                _, corrupted_labels = torch.max(corrupted_out, 1)
                                true_classified = torch.eq(corrupted_labels, detected_labels)
                                analysis_counters_weights[11] += 1
                                index_tensor = torch.logical_not(true_classified) * (index_tensor - 1) + true_classified * (index_tensor + 1)
                                index_tensor[index_tensor >= vector_len] = vector_len - 1
                                index_tensor[index_tensor < 0] = 0
                                handle.remove()
                            dlt_search_h = VVSS_dict_weights['pos_inf'][index_tensor]
                            dlt_search_h[grad_bool_map == 1] = inf_represent 

                    else:
                        analysis_counters_weights[9] += 1           #mixed_positive
                        vector_len_inf = VVSS_dict_weights['pos_inf'].size(0)
                        vector_len_1 = VVSS_dict_weights['pos_1'].size(0)
                        iteration_count_inf = vector_len_inf // 2
                        iteration_count_1 = vector_len_1 // 2     #will be bigger value
                        index_tensor_inf = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_inf
                        index_tensor_1 = torch.ones(batch_size, dtype=torch.int, device=self.device) * iteration_count_1
                        true_classified_init = torch.clone(true_classified)
                        for _ in range(iteration_count_1, 0, -1):
                            dlt_search_h = torch.logical_not(true_classified_init) * VVSS_dict_weights['pos_1'][index_tensor_1] + true_classified_init * VVSS_dict_weights['pos_inf'][index_tensor_inf]
                            handle = layer.register_forward_hook(self.delta_injection_channel(rand_channel, dlt_search_h))
                            corrupted_out = self.nn_module(images)
                            _, corrupted_labels = torch.max(corrupted_out, 1)
                            true_classified = torch.eq(corrupted_labels, detected_labels)
                            analysis_counters_weights[11] += 1
                            index_tensor_inf = true_classified_init * (torch.logical_not(true_classified) * (index_tensor_inf - 1) + true_classified * (index_tensor_inf + 1))
                            index_tensor_1 = torch.logical_not(true_classified_init) * (torch.logical_not(true_classified) * (index_tensor_1 - 1) + true_classified * (index_tensor_1 + 1))
                            index_tensor_inf[index_tensor_inf >= vector_len_inf] = vector_len_inf - 1
                            index_tensor_inf[index_tensor_inf < 0] = 0
                            index_tensor_1[index_tensor_1 >= vector_len_1] = vector_len_1 - 1
                            index_tensor_1[index_tensor_1 < 0] = 0
                            handle.remove()
                        dlt_search_h = torch.logical_not(true_classified_init) * VVSS_dict_weights['pos_1'][index_tensor_1] + true_classified_init * VVSS_dict_weights['pos_inf'][index_tensor_inf]
                        dlt_search_h[grad_bool_map == 1] = inf_represent        # VVSS_dict_weights['pos_inf'][-1]
                
                else:
                    analysis_counters_weights[10] += 1           #skipped_positive
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
                
                non_crit_channels[rand_channel] = noncriticality_channel / batch_size

                #free memory
                del self.activation.grad
                del loss_deepvigor
                del channel_grad 
                del grad_bool_map
                del corrupted_out
                del neurons_set
                torch.cuda.empty_cache()

            torch.cuda.empty_cache()
            sampled_LVF = torch.sum(non_crit_channels) / ch_counter
            
        return non_crit_channels, analysis_counters_weights, sampled_LVF


    