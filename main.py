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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model-level CNN hardening")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "imagenet"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--is-pruning", type=bool, default=False)
    parser.add_argument("--pruning-method", type=str, default="hm")
    parser.add_argument("--pruning-ratio", type=float, default=0.0)
    parser.add_argument("--is-hardening", type=bool, default=False)
    parser.add_argument("--hardening-ratio", type=float, default=0.0)
    parser.add_argument("--importance", type=str, choices=["l1-norm", "vul-gain", "salience"], default="")


    # setting up the values
    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    batch_size = args.batch_size
    is_pruning = args.is_pruning
    pruning_method = args.pruning_method
    pruning_ratio = args.pruning_ratio
    is_hardening = args.is_hardening
    hardening_ratio = args.hardening_ratio
    importance_command = args.importance 

    # create log file
    run_mode = "test"
    run_mode += " ".join([part for part, condition in [("_pruning", is_pruning), ("_hardening", is_hardening)] if condition])
    setup = handlers.LogHandler(run_mode, model_name, dataset_name)   
    logger = setup.getLogger()

    # set the device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # load dataset and CNN model
    trainloader, classes_count, dummy_input = dataset_utils.load_dataset(dataset_name, batch_size, is_train=True)
    testloader, classes_count, dummy_input = dataset_utils.load_dataset(dataset_name, batch_size, is_train=False)
    model = models_utils.load_model(model_name, dataset_name, device)
    dummy_input = dummy_input.to(device)

    runModeHandler = handlers.RunModeHandler(logger)
    runModeHandler.register("test", run_mode_utils.test)
    runModeHandler.register("test_pruning", run_mode_utils.pruning_func)
    runModeHandler.register("test_hardening", run_mode_utils.hardening_func)

    if run_mode == "test":
        runModeHandler.execute(run_mode, model, testloader, device, dummy_input, logger)
    
    elif run_mode == "test_pruning":
        runModeHandler.execute(run_mode, model, trainloader, testloader, classes_count, dummy_input, 
                               pruning_method, device, pruning_ratio, importance_command, logger)
    
    elif run_mode == "test_hardening":
        runModeHandler.execute(run_mode, model, trainloader, testloader, dummy_input, classes_count, 
                               pruning_method, hardening_ratio, importance_command, device, logger)
    

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


    #TODO: pruning with non-unified pruning ratio + refining
    #TODO: iterative pruning + refining

