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
    parser.add_argument("--is-pruning", action='store_true')
    parser.add_argument("--is-pruned", action='store_true')
    parser.add_argument("--pruning-method", type=str, default="hm")
    parser.add_argument("--pruning-ratio", type=float, default=None)
    parser.add_argument("--pruned-checkpoint", type=str, default=None)
    parser.add_argument("--is-hardening", action='store_true')
    parser.add_argument("--is-hardened", action='store_true')
    parser.add_argument("--hardening-ratio", type=float, default=None)
    parser.add_argument("--hardened-checkpoint", type=str, default=None)
    parser.add_argument("--importance", type=str, choices=["l1-norm", "vul-gain", "salience"], default=None)


    # setting up the arguments values
    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    batch_size = args.batch_size
    is_pruning = args.is_pruning
    is_pruned = args.is_pruned
    pruning_method = args.pruning_method
    pruning_ratio = args.pruning_ratio
    pruned_checkpoint = args.pruned_checkpoint
    is_hardening = args.is_hardening
    is_hardened = args.is_hardened
    hardening_ratio = args.hardening_ratio
    hardened_checkpoint = args.hardened_checkpoint
    importance_command = args.importance 

    #is_hardening and is_pruning should not be True at a same time
    assert not (is_hardening and is_pruning) == True

    # create log file
    run_mode = "test"
    run_mode += "".join([part for part, condition in [("_pruning", is_pruning), ("_hardening", is_hardening)] if condition])
    setup = handlers.LogHandler(run_mode, model_name, dataset_name)   
    logger = setup.getLogger()
    setup_logger_info = ""
    setup_logger_info += "".join(f"{i}: {args.__dict__[i]}, " for i in args.__dict__ if \
                                    (type(args.__dict__[i]) is bool and args.__dict__[i] is True) or \
                                    (type(args.__dict__[i]) is not bool and args.__dict__[i] is not None))
    logger.info(f"args: {setup_logger_info}")

    # set the device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # load dataset and CNN model
    trainloader, classes_count, dummy_input = dataset_utils.load_dataset(dataset_name, batch_size, is_train=True)
    testloader, classes_count, dummy_input = dataset_utils.load_dataset(dataset_name, batch_size, is_train=False)
    model = models_utils.load_model(model_name, dataset_name, device)

    if is_pruned:
        assert pruning_ratio is not None
        assert pruned_checkpoint is not None
        pu = utils.prune_utils(model, pruning_method, classes_count, pruning_method)
        pu.set_pruning_ratios(pruning_ratio)
        model = pu.homogeneous_prune(model)
        models_utils.load_params(model, pruned_checkpoint, device)

    if is_hardened:
        assert hardening_ratio is not None
        assert hardened_checkpoint is not None
        hr = utils.hardening_utils(hardening_ratio)
        model = hr.conv_replacement(model)
        models_utils.load_params(model, hardened_checkpoint, device)


    dummy_input = dummy_input.to(device)

    runModeHandler = handlers.RunModeHandler(logger)
    runModeHandler.register("test", run_mode_utils.test_func)
    runModeHandler.register("test_pruning", run_mode_utils.pruning_func)
    runModeHandler.register("test_hardening", run_mode_utils.hardening_func)

    if run_mode == "test":
        runModeHandler.execute(run_mode, model, testloader, device, dummy_input, logger)
    
    elif run_mode == "test_pruning":
        assert importance_command is not None
        assert pruning_ratio is not None
        runModeHandler.execute(run_mode, model, trainloader, testloader, classes_count, dummy_input, 
                               pruning_method, device, pruning_ratio, importance_command, logger)
    
    elif run_mode == "test_hardening":
        assert importance_command is not None
        assert hardening_ratio is not None
        runModeHandler.execute(run_mode, model, trainloader, testloader, dummy_input, classes_count, 
                               pruning_method, hardening_ratio, importance_command, device, logger)
    

    #TODO: pruning with non-unified pruning ratio + refining
    #TODO: iterative pruning + refining

