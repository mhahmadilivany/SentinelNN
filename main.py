import torch
import time
import sys
import getopt
import os
import dataset_utils
import models_utils
import torchprofile




if __name__ == "__main__":
    arg_list = sys.argv[1:]
    short_options = "m:b:d:r"
    long_options = ["model=", "batch-size=", "dataset=", "run-mode="]
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
            elif arg in ["r", "--run-mode"]:
                run_mode = str(val)

    except:
        raise Exception("parameters are not specified correctly!")


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    
    #load dataset and CNN model
    is_train = False
    dataloader, classes_count, dummy_input = dataset_utils.load_dataset(dataset_name, is_train, batch_size)
    model = models_utils.load_model(network_name, dataset_name, device)
    
    #create log file
    log_direction = network_name + "-" + dataset_name + "/" + run_mode
    if not os.path.exists(log_direction):
        os.makedirs(log_direction)
    log_file_name = log_direction + "/log-" + run_mode + "-" + network_name + "-" + dataset_name + ".txt"
    log_file = open(log_file_name, 'w')
    log_file.close()

    
    if run_mode == "test":
        net_accuracy = models_utils.evaluate(model, dataloader, device=device)
        total_params = sum(p.numel() for p in model.parameters())
        total_macs = torchprofile.profile_macs(model, dummy_input)
        log_file = open(log_file_name, 'w')
        log_file.write("model test top-1 accuracy: " + str(net_accuracy) + "\n")
        log_file.write("total number of MACs: " + str(total_macs) + "\n") 
        log_file.write("total number of parameters: " + str(total_params) + "\n")
        log_file.close()
        print("test done successfuly!")
        

        

    """for data in dataloader:
        images, labels = data[0].to(device), data[1].to(device)

        #TODO: forward evaluation of the model

        #TODO: pruning the model with various sensetivity analysis methods
        # and unified pruning ratio + refining 

        #TODO: model modification for channel duplication
        
        #TODO: pruning with non-unified pruning ratio + refining
        #TODO: iterative pruning + refining

        #if run_mode == "something":


        break"""