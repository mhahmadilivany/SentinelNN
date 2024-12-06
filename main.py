import torch
import time
import sys
import getopt
import argparse
import os
import dataset_utils
import models_utils
import torchprofile
import utils
import copy
import torch.nn as nn
import pruning
import importance_analysis as imp
import run_mode_utils
import handlers
from typing import Union



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model-level CNN hardening")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "imagenet"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--is-pruned", type=bool, default=False)
    parser.add_argument("--pruning-method", type=str, default="hm")
    parser.add_argument("--pruning-ratio", type=float, default=0)
    parser.add_argument("--is-hardened", type=bool, default=False)
    parser.add_argument("--hardening-ratio", type=float, default=0)
    parser.add_argument("--importance", type=str, choices=["l1-norm", "vul-gain", "salience"], default="")


    # setting up the values
    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    batch_size = args.batch_size
    is_pruned = args.is_pruned
    pruning_method = args.pruning_method
    pruning_ratio = args.pruning_ratio
    is_hardened = args.is_hardened
    hardening_ratio = args.hardening_ratio
    importance_command = args.importance 
    #print(pruning_ratio, pruning_ratio[0])

    # create log file
    run_mode = "test"
    run_mode += " ".join([part for part, condition in [("_pruning", is_pruned), ("_hardening", is_hardened)] if condition])
    setup = handlers.LogHandler(run_mode, model_name, dataset_name)   
    logger = setup.getLogger()

    # set the device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # load dataset and CNN model
    trainloader, classes_count, dummy_input = dataset_utils.load_dataset(dataset_name, batch_size, is_train=True)
    testloader, classes_count, dummy_input = dataset_utils.load_dataset(dataset_name, batch_size, is_train=False)
    model = models_utils.load_model(model_name, dataset_name, device)
    dummy_input = dummy_input.to(device)

    runModeHandler = handlers.RunModeHandler()
    runModeHandler.register("test", run_mode_utils.test)
    runModeHandler.register("test_pruning", run_mode_utils.pruning_func)

    if run_mode == "test":
        runModeHandler.execute(run_mode, logger, model, testloader, device, dummy_input, logger)
    elif run_mode == "test_pruning":
        runModeHandler.execute(run_mode, logger, model, trainloader, testloader, classes_count, dummy_input, 
                               pruning_method, device, pruning_ratio, importance_command, logger)
    # elif run_mode == "test_hardening":
    #     runModeHandler.execute(run_mode, logger, ...)
    

    # loads pruned models
    if run_mode == "load-pruned":
        pruned_file_name = log_direction + '/../pruned_model-' + str(pruning_ratio) + '.pth'
        model_cp = copy.deepcopy(model)
        pu = utils.prune_utils(model_cp, pruning_method)
        pu.set_pruning_ratios(pruning_ratio)
        pruned_model = pu.homogeneous_prune(model_cp)

        models_utils.load_params(pruned_model, pruned_file_name, device)
        
        model_accuracy = models_utils.evaluate(pruned_model, testloader, device)
        model_params = sum(p.numel() for p in pruned_model.parameters())
        model_macs = torchprofile.profile_macs(pruned_model, dummy_input)
        print(model_accuracy, model_params, model_macs)
    
    # model modification for channel duplication
    # TODO: saving and loading them 
    
    elif run_mode == "hardening":
        handler = handlers.AnalysisHandler()
        
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

