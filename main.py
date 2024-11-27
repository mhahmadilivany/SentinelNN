import torch
import time
import sys
import getopt
import os
import dataset_utils
import models_utils
import torchprofile
import utils
import copy
import torch.nn as nn
import pruning
import importance_analysis as vul



if __name__ == "__main__":
    arg_list = sys.argv[1:]
    short_options = "m:b:d:r"
    long_options = ["model=", "batch-size=", "dataset=", "run-mode=", "pruning-method=", "pruning-list=", "importance="]
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
            elif arg in ["-r", "--run-mode"]:
                run_mode = str(val)
            elif arg in ["-h", "--pruning-method"]:
                pruning_method = str(val)
            elif arg in ["-p", "--pruning-list"]:
                pruning_vals = str(val)
                pruning_list = pruning_vals.split(',')
                pruning_ratio_list = []
                for item in pruning_list:
                    pruning_ratio_list.append(float(item))
            elif arg in ["-i", "--importance"]:
                importance_command = str(val)
    except:
        raise Exception("parameters are not specified correctly!")


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    
    #load dataset and CNN model
    trainloader, classes_count, dummy_input = dataset_utils.load_dataset(dataset_name, batch_size, is_train=True)
    testloader, classes_count, dummy_input = dataset_utils.load_dataset(dataset_name, batch_size, is_train=False)
    model = models_utils.load_model(network_name, dataset_name, device)
    dummy_input = dummy_input.to(device)
    
    #create log file
    log_direction = network_name + "-" + dataset_name + "/" + run_mode
    if not os.path.exists(log_direction):
        os.makedirs(log_direction)
    log_file_name = log_direction + "/log-" + run_mode + "-" + network_name + "-" + dataset_name + ".txt"
    log_file = open(log_file_name, 'w')
    log_file.close()

    if run_mode == "test":
        net_accuracy = models_utils.evaluate(model, testloader, device=device)
        total_params = sum(p.numel() for p in model.parameters())
        total_macs = torchprofile.profile_macs(model, dummy_input)
        log_file = open(log_file_name, 'w')
        log_file.write("model test top-1 accuracy: " + str(net_accuracy) + "\n")
        log_file.write("total number of MACs: " + str(total_macs) + "\n") 
        log_file.write("total number of parameters: " + str(total_params) + "\n")
        log_file.close()
        print("test done successfuly!")
        
    #create prune_utils for the related functions
    #develop based on apply_channel_sorting
    #have a separate function for resilience analysis, replaceable by different functions
    #apply channel sorting
    #recursive iteration over the CNN
    #apply one prune ratio for conv layers and one for FC
    #another function for refining which includes regularization as well
    #save and load the pruned CNN 
    elif run_mode == "pruning":
        handler = utils.AnalysisHandler()
        
        #registering commands for importance analysis 
        handler.register("l1-norm", vul.L1_norm)
        handler.register("vul-gain", vul.vulnerability_gain)

        model_cp = copy.deepcopy(model)
        pu = utils.prune_utils(model_cp, trainloader, classes_count, pruning_method, device)
        pu.set_pruning_ratios(pruning_ratio_list)

        sorted_model = pu.channel_sorting(model_cp, handler, importance_command)
        
        pruned_model = pu.homogeneous_prune(sorted_model)
        print(pruned_model)
        
        model_accuracy = models_utils.evaluate(model, testloader, device=device)
        model_params = sum(p.numel() for p in model.parameters())
        model_macs = torchprofile.profile_macs(model, dummy_input)

        pruned_accuracy = models_utils.evaluate(pruned_model, testloader, device=device)
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        pruned_macs = torchprofile.profile_macs(pruned_model, dummy_input)
        print("pruned model test top-1 accuracy: " + str(pruned_accuracy))

        #fine tuning the pruned model
        finetune_epochs = 10
        finetune_model = pruning.fine_tune(pruned_model, trainloader, testloader, finetune_epochs, device)
        finetune_accuracy = models_utils.evaluate(finetune_model, testloader, device=device)
        
        torch.save(finetune_model.state_dict(), log_direction + '/../pruned_model-' + pruning_vals + '.pth')

        print("fine tuned model test top-1 accuracy: " + str(finetune_accuracy))
        print("pruned model number of MACs: " + str(pruned_macs)) 
        print("pruned model number of parameters: " + str(pruned_params))
        print("final accuracy loss: " + str(model_accuracy - finetune_accuracy))
        print("MAC improvement: " + str(pruned_params / model_params))
        print("Params improvement: " + str(pruned_macs / model_macs))


    elif run_mode == "load-pruned":
        pruned_file_name = log_direction + '/../pruned_model-' + pruning_vals + '.pth'
        model_cp = copy.deepcopy(model)
        pu = utils.prune_utils(model_cp, pruning_method)
        pu.set_pruning_ratios(pruning_ratio_list)
        pruned_model = pu.homogeneous_prune()

        models_utils.load_params(pruned_model, pruned_file_name, device)
        
        model_accuracy = models_utils.evaluate(pruned_model, testloader, device)
        model_params = sum(p.numel() for p in pruned_model.parameters())
        model_macs = torchprofile.profile_macs(pruned_model, dummy_input)
        print(model_accuracy, model_params, model_macs)
        
    
    elif run_mode == "vul-analysis":
        handler = utils.analysis_handler()
        
        #registering commands for importance analysis 
        handler.register("l1-norm", pruning.channel_L1_norm)
        handler.register("vuln-gain", vul.channels_vulnerability)

        if importance_command == "l1-norm":
            handler.execute(importance_command, )
    
    
    #TODO: model modification for channel duplication
    #add correction_utils for related functions
    #develop a class instead of Conv2d, to replace them with it
    #it includes duplication and correction and outputs the same expected size
    #saving and loading them 

    #TODO: pruning with non-unified pruning ratio + refining
    #TODO: iterative pruning + refining

    