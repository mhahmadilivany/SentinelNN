import torch
import binary_converter


def creating_VVSS_dict(errors_distribution, resolution, device):
    dict_errors_distribution = {}
    vul = -2 ** resolution
    ind = 0
    for _ in range(4 * resolution + 3):      #+3 represnets points for -inf, 0, +inf
        dict_errors_distribution[vul] = errors_distribution[ind].item()

        #update vul
        vul = (vul < 0) * vul / 2 + (vul > 0) * vul * 2 + (vul == 0) * (2**(-resolution))
        if vul < 0 and vul > -2 ** (-resolution):
            vul = 0
        elif vul > 2 ** resolution:
            break
        ind += 1
    
    VVSS_dict = {'neg_inf': [], 'neg_1': [], 'pos_1': [], 'pos_inf': []}
    prev_val = 0
    for i in dict_errors_distribution:
        if i < 0 and dict_errors_distribution[i] - prev_val != 0:
            if i < -1:
                VVSS_dict['neg_inf'].append(i)
            elif i >= -1:
                VVSS_dict['neg_1'].append(i)

        elif i > 0 and dict_errors_distribution[i] - prev_val != 0:
            if i <= 1:
                VVSS_dict['pos_1'].append(i)
            elif i > 1 and i < 2 ** resolution + 1:
                VVSS_dict['pos_inf'].append(i)
        prev_val = dict_errors_distribution[i]

    if 2 ** resolution not in VVSS_dict['pos_inf']:
        VVSS_dict['pos_inf'].append(2 ** resolution)
    if -2 ** resolution not in VVSS_dict['neg_inf']:
        VVSS_dict['neg_inf'].append(-2 ** resolution)


    VVSS_dict['neg_inf'] = torch.tensor(VVSS_dict['neg_inf'], dtype=torch.float).to(device)
    VVSS_dict['neg_1'] = torch.tensor(VVSS_dict['neg_1'], dtype=torch.float).to(device)
    VVSS_dict['pos_1'] = torch.tensor(VVSS_dict['pos_1'], dtype=torch.float).to(device)
    VVSS_dict['pos_inf'] = torch.tensor(VVSS_dict['pos_inf'], dtype=torch.float).to(device)

    return VVSS_dict


def vulnerability_values_grouping(errors_list, resolution):
    #resolution = 10
    vul = -2 ** resolution
    errors_count = torch.numel(errors_list)
    vul_dict = {}
    
    for _ in range(4 * resolution + 3):      #+3 represnets points for -inf, 0, +inf
        dist = torch.sum(errors_list < vul) / errors_count 
        vul_dict[vul] = dist.item()

        #update vul
        vul = (vul < 0) * vul / 2 + (vul > 0) * vul * 2 + (vul == 0) * (2**(-resolution))
        if vul < 0 and vul > -2 ** (-resolution):
            vul = 0
        elif vul > 2 ** resolution:
            break

    return vul_dict


def vulnerability_values_space_weight(inputs, weights, device):     #obtaining added errors to conv's output with SEU in weights
    deviations = torch.zeros((32, torch.numel(inputs)), device=device)
    resolution = 10

    weights_cp = torch.clone(weights)
    for bit_loc in range(32):
        weights_bin = binary_converter.float2bit(weights_cp.flatten(), device=device)
        weights_bin[:, bit_loc] = torch.logical_xor(weights_bin[:, bit_loc], torch.ones(weights_bin[:, bit_loc].size(), device=device)).float()
        dev = binary_converter.bit2float(weights_bin, device=device) - weights.flatten()
        dev = dev.reshape(weights.size()) * inputs
        dev = torch.nan_to_num(dev, nan=10000)
        deviations[bit_loc] = dev.flatten()
        
        weights_cp = torch.clone(weights)
    
    dict_errors_dist_weight = vulnerability_values_grouping(torch.sort(deviations.flatten())[0], resolution)

    errors_dist_weight = torch.zeros(len(dict_errors_dist_weight), device=device)
    ind = 0
    for i in dict_errors_dist_weight:
        errors_dist_weight[ind] = dict_errors_dist_weight[i]
        ind += 1

    #free up memory
    del weights_cp
    del weights_bin
    del dev
    del deviations
    del dict_errors_dist_weight
    torch.cuda.empty_cache()
    
    return errors_dist_weight
