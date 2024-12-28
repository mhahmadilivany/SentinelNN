import torch
import time
import sys
import getopt
import os
import dataset_utils
import analysis_utils
import CNNs


if __name__ == "__main__":
    arg_list = sys.argv[1:]
    short_options = "m:c:b:d:r"
    long_options = ["model=", "channel-sampling=", "batch-size=", "dataset=", "run-mode="]
    try:
        arguments, values = getopt.getopt(arg_list, short_options, long_options)
        for arg, val in arguments:
            print(arg, val)
            if arg in ["-m", "--model"]:
                network_name = str(val)
            elif arg in ["-d", "--dataset"]:
                dataset_name = str(val)
            elif arg in ["-b", "--batch-size"]:
                batch_size = int(val)
            elif arg in ["r", "--run-mode"]:        #"sampling-analysis", "full-analysis-weight", "sampling-analysis-act", "sampling-analysis-weight"
                run_mode = str(val)
            elif arg in ["c", "--channel-sampling"]:
                ch_sampling_ratio = float(val)

    except:
        raise Exception("parameters are not specified correctly!")
    
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    #load dataset and CNN model
    is_train = False
    dataloader, classes_count = dataset_utils.load_dataset(dataset_name, is_train, batch_size)
    net = CNNs.load_model(network_name, device)
    
    #create log file
    log_direction = network_name + "-" + dataset_name + "/" + run_mode
    if not os.path.exists(log_direction):
        os.makedirs(log_direction)
    log_file_name = log_direction + "/log-" + run_mode + "-" + network_name + "-" + dataset_name + ".txt"
    log_file = open(log_file_name, 'w')
    log_file.close()

    #performing based on run_mode
    for data in dataloader:
        images, labels = data[0].to(device), data[1].to(device)

        if run_mode == "full-analysis-weight":
            au = analysis_utils.analysis_utils(net, network_name, images, dataset_name, batch_size, classes_count, log_file_name, log_direction, device)
            au.full_weights_analysis(net)
            
        elif run_mode == "full-analysis-act":
            au = analysis_utils.analysis_utils(net, network_name, images, dataset_name, batch_size, classes_count, log_file_name, log_direction, device)
            au.full_activations_analysis(net)
        
        elif run_mode == "sampling-analysis-act":
            log_file_name = log_direction + "/log-" + run_mode + "-" + network_name + "-" + dataset_name + "-" + str(ch_sampling_ratio) + ".txt"
            log_file = open(log_file_name, 'w')
            log_file.close()
            
            au = analysis_utils.analysis_utils(net, network_name, images, dataset_name, batch_size, classes_count, log_file_name, log_direction, device, channel_sampling_ratio=ch_sampling_ratio)
            au.sampling_analysis_act(net)
        
        elif run_mode == "sampling-analysis-weight":
            log_file_name = log_direction + "/log-" + run_mode + "-" + network_name + "-" + dataset_name + "-" + str(ch_sampling_ratio) + ".txt"
            log_file = open(log_file_name, 'w')
            log_file.close()
            
            au = analysis_utils.analysis_utils(net, network_name, images, dataset_name, batch_size, classes_count, log_file_name, log_direction, device, channel_sampling_ratio=ch_sampling_ratio)
            au.sampling_analysis_weight(net)

        break



