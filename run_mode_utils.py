import torch
import torch.nn as nn
import models_utils
import torchprofile
import importance_analysis as imp
import copy
import utils
import pruning
from torch.utils.data import DataLoader
from typing import Union
import logging
import handlers

def test(model: nn.Module, 
         testloader: DataLoader, 
         device: Union[torch.device, str], 
         dummy_input: torch.tensor, 
         logger: logging.Logger) -> Union[float, int, int]:
    
    net_accuracy = models_utils.evaluate(model, testloader, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    total_macs = torchprofile.profile_macs(model, dummy_input)
    logger.info(f"model test top-1 accuracy: {net_accuracy}%")
    logger.info(f"total number of MACs: {total_macs}")
    logger.info(f"total number of parameters: {total_params}")

    return net_accuracy, total_params, total_macs


def pruning_func(model: nn.Module, 
                 trainloader: DataLoader, 
                 testloader: DataLoader, 
                 classes_count: int,
                 dummy_input: torch.tensor, 
                 pruning_method: str, 
                 device: Union[torch.device, str], 
                 pruning_ratio: Union[float, list], 
                 importance_command: str, 
                 logger: logging.Logger) -> None:

    model_accuracy, model_params, model_macs = test(model, testloader, device, dummy_input, logger)
    
    handler = handlers.AnalysisHandler(logger)
        
    # registering commands for importance analysis 
    handler.register("l1-norm", imp.L1_norm)
    handler.register("vul-gain", imp.vulnerability_gain)
    handler.register("salience", imp.Salience)

    model_cp = copy.deepcopy(model)
    pu = utils.prune_utils(model_cp, trainloader, classes_count, pruning_method, device)
    pu.set_pruning_ratios(pruning_ratio)

    sorted_model = pu.channel_sorting(model_cp, handler, importance_command)
    logger.info("channels are sorted")

    pruned_model = pu.homogeneous_prune(sorted_model)
    logger.info("model is pruned")

    pruned_accuracy = models_utils.evaluate(pruned_model, testloader, device=device)
    pruned_params, pruned_macs = models_utils.size_profile(pruned_model, dummy_input)
    logger.info(f"pruned model test top-1 accuracy: {pruned_accuracy}%")

    #fine tuning the pruned model and saves the best accuracy
    finetune_epochs = 10
    finetune_model = pruning.fine_tune(pruned_model, trainloader, testloader, finetune_epochs, device, logger, str(pruning_ratio))
    models_utils.load_params(finetune_model, f'./../pruned_model-{pruning_ratio}.pth', device)       #loads the model which achieved best accuracy
    finetune_accuracy = models_utils.evaluate(finetune_model, testloader, device=device)

    logger.info(f"fine tuned pruned model test top-1 accuracy: {finetune_accuracy}%")
    logger.info(f"pruned model number of MACs: {pruned_macs}") 
    logger.info(f"pruned model number of parameters: {pruned_params}")
    logger.info(f"final accuracy loss: {model_accuracy - finetune_accuracy}%")
    logger.info(f"MAC improvement: {pruned_params / model_params}")
    logger.info(f"Params improvement: {pruned_macs / model_macs}")

    
def hardening_func(model: nn.Module,
                   trainloader: DataLoader,
                   testloader: DataLoader,
                   dummy_input: torch.tensor,
                   classes_count: int,
                   pruning_method: str,
                   hardening_ratio: float,
                   importance_command: str,
                   device: Union[torch.device, str],
                   logger: logging.Logger) -> None:
    
    handler = handlers.AnalysisHandler(logger)
        
    #registering commands for importance analysis 
    handler.register("l1-norm", imp.L1_norm)
    handler.register("vul-gain", imp.vulnerability_gain)
    handler.register("salience", imp.Salience)

    _, model_params, model_macs = test(model, testloader, device, dummy_input, logger)

    model_cp = copy.deepcopy(model)
    pu = utils.prune_utils(model_cp, trainloader, classes_count, pruning_method, device)

    sorted_model = pu.channel_sorting(model_cp, handler, importance_command)
    logger.info("channels are sorted")
    
    hr = utils.hardening_utils(hardening_ratio)
    hardened_model = hr.conv_replacement(sorted_model)  #replace all Conv2d with HardenedConv2d
    log_dir = logger.handlers[0].baseFilename.split("log")[0]
    torch.save(hardened_model.state_dict(), f'{log_dir}/../hardened_model-{hardening_ratio}.pth')
    logger.info("model is hardened and saved")

    hardened_accuracy, hardened_params, hardened_macs = test(hardened_model, testloader, device, dummy_input, logger)
    
    logger.info(f"model test top-1 accuracy: {hardened_accuracy}%")
    logger.info(f"MACs overhead: {hardened_params / model_params}")
    logger.info(f"Params overhead: {hardened_macs / model_macs}")