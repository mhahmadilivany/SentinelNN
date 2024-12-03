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
import importance_analysis as imp



if __name__ == "__main__":
    arg_list = sys.argv[1:]
    short_options = "m:b:d:r"
    long_options = ["run-mode=", "model=", "batch-size=", "dataset=", "pruning-method=", "pruning-list=", "importance=", "hardening-ratio="]
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
            elif arg in ["-u", "--pruning-method"]:
                pruning_method = str(val)
            elif arg in ["-p", "--pruning-list"]:
                pruning_vals = str(val)
                pruning_list = pruning_vals.split(',')
                pruning_ratio_list = []
                for item in pruning_list:
                    pruning_ratio_list.append(float(item))
            elif arg in ["-i", "--importance"]:
                importance_command = str(val)
            elif arg in ["-h", "--hardening-ratio"]:
                hardening_ratio = float(val)
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

    # examines the accuracy of CNNs
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
        
    # prunes Conv2d layers based on the assigned pruning ratio and importance analysis
    elif run_mode == "pruning":
        handler = utils.AnalysisHandler()
        
        #registering commands for importance analysis 
        handler.register("l1-norm", imp.L1_norm)
        handler.register("vul-gain", imp.vulnerability_gain)
        handler.register("salience", imp.Salience)

        model_cp = copy.deepcopy(model)
        pu = utils.prune_utils(model_cp, trainloader, classes_count, pruning_method, device)
        pu.set_pruning_ratios(pruning_ratio_list)

        sorted_model = pu.channel_sorting(model_cp, handler, importance_command)
        pruned_model = pu.homogeneous_prune(sorted_model)
        
        model_accuracy = models_utils.evaluate(model, testloader, device=device)
        model_params, model_macs = models_utils.size_profile(model, dummy_input)

        pruned_accuracy = models_utils.evaluate(pruned_model, testloader, device=device)
        pruned_params, pruned_macs = models_utils.size_profile(pruned_model, dummy_input)
        print("pruned model test top-1 accuracy: " + str(pruned_accuracy))

        #fine tuning the pruned model and saves the best accuracy
        finetune_epochs = 10
        finetune_model = pruning.fine_tune(pruned_model, trainloader, testloader, finetune_epochs, device, log_direction, pruning_vals)
        models_utils.load_params(finetune_model, log_direction + '/../pruned_model-' + pruning_vals + '.pth', device)       #loads the model which achieved best accuracy
        finetune_accuracy = models_utils.evaluate(finetune_model, testloader, device=device)

        print("fine tuned model test top-1 accuracy: " + str(finetune_accuracy))
        print("pruned model number of MACs: " + str(pruned_macs)) 
        print("pruned model number of parameters: " + str(pruned_params))
        print("final accuracy loss: " + str(model_accuracy - finetune_accuracy))
        print("MAC improvement: " + str(pruned_params / model_params))
        print("Params improvement: " + str(pruned_macs / model_macs))

    # loads pruned models
    elif run_mode == "load-pruned":
        pruned_file_name = log_direction + '/../pruned_model-' + pruning_vals + '.pth'
        model_cp = copy.deepcopy(model)
        pu = utils.prune_utils(model_cp, pruning_method)
        pu.set_pruning_ratios(pruning_ratio_list)
        pruned_model = pu.homogeneous_prune(model_cp)

        models_utils.load_params(pruned_model, pruned_file_name, device)
        
        model_accuracy = models_utils.evaluate(pruned_model, testloader, device)
        model_params = sum(p.numel() for p in pruned_model.parameters())
        model_macs = torchprofile.profile_macs(pruned_model, dummy_input)
        print(model_accuracy, model_params, model_macs)
        


    # model modification for channel duplication
    # TODO: saving and loading them 
    elif run_mode == "hardening":
        handler = utils.AnalysisHandler()
        
        #registering commands for importance analysis 
        handler.register("l1-norm", imp.L1_norm)
        handler.register("vul-gain", imp.vulnerability_gain)
        handler.register("salience", imp.Salience)
        t_tmp = time.time()
        model_accuracy = models_utils.evaluate(model, testloader, device=device)
        model_time = time.time() - t_tmp
        model_params, model_macs = models_utils.size_profile(model, dummy_input)
        print(model_accuracy)

        model_cp = copy.deepcopy(model)
        pu = utils.prune_utils(model_cp, trainloader, classes_count, pruning_method, device)
        pu.set_pruning_ratios(pruning_ratio_list)

        sorted_model = pu.channel_sorting(model_cp, handler, importance_command)
        
        hr = utils.hardening_utils(hardening_ratio)
        hardened_model = hr.conv_replacement(sorted_model) #replace all Conv2d with HardenedConv2d
        t_tmp = time.time()
        hardened_accuracy = models_utils.evaluate(hardened_model, testloader, device=device)
        hardened_time = time.time() - t_tmp
        hardened_params, hardened_macs = models_utils.size_profile(hardened_model, dummy_input)
        
        print(hardened_accuracy)
        print("MACs overhead: " + str(hardened_params / model_params))
        print("Params overhead: " + str(hardened_macs / model_macs))
        print("Performance overhead: " + str(hardened_time / model_time))

    


    #TODO: pruning with non-unified pruning ratio + refining
    #TODO: iterative pruning + refining



    