import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import models_utils
import torchprofile
import importance_analysis as imp
import copy
import utils
import pruning
from typing import Union
import logging
import handlers
import clipping
import fault_simulation

def test_func(model: nn.Module, 
         testloader: DataLoader, 
         device: Union[torch.device, str], 
         dummy_input: torch.tensor, 
         logger: logging.Logger) -> Union[float, int, int]:
    model.eval()
    with torch.no_grad():
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

    model_accuracy, model_params, model_macs = test_func(model, testloader, device, dummy_input, logger)
    
    handler = handlers.AnalysisHandler(logger)
        
    # registering commands for importance analysis 
    handler.register("l1-norm", imp.L1_norm)
    handler.register("vul-gain", imp.vulnerability_gain)
    handler.register("salience", imp.Salience)

    #model_cp = copy.deepcopy(model)
    pu = utils.prune_utils(model, trainloader, classes_count, pruning_method, device)
    pu.set_pruning_ratios(pruning_ratio)

    sorted_model = pu.channel_sorting(model, handler, importance_command)
    logger.info("channels are sorted")
    model_accuracy, model_params, model_macs = test_func(model, testloader, device, dummy_input, logger)

    pruned_model = pu.homogeneous_prune(sorted_model)
    logger.info(f"model is pruned: {pruned_model}")

    pruned_accuracy = models_utils.evaluate(pruned_model, testloader, device=device)
    pruned_params, pruned_macs = models_utils.size_profile(pruned_model, dummy_input)
    logger.info(f"pruned model test top-1 accuracy: {pruned_accuracy}%")
    logger.info(f"pruned model number of MACs: {pruned_macs}") 
    logger.info(f"pruned model number of parameters: {pruned_params}")

    #fine tuning the pruned model and saves the best accuracy
    finetune_epochs = 10
    finetune_model = pruning.fine_tune(pruned_model, trainloader, testloader, finetune_epochs, device, logger, pruning_ratio, importance_command)
    log_dir = logger.handlers[0].baseFilename.split("log")[0]
    models_utils.load_params(finetune_model, f'{log_dir}/../pruned_model-{importance_command}-{pruning_ratio}.pth', device)       #loads the model which achieved best accuracy
    finetune_accuracy = models_utils.evaluate(finetune_model, testloader, device=device)

    logger.info(f"fine tuned pruned model test top-1 accuracy: {finetune_accuracy}%")
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
                   clipping_command: str,
                   device: Union[torch.device, str],
                   logger: logging.Logger) -> None:
    
    # Creating handler for ReLU clipping 
    # and registering commands for it
    clippingHandler = handlers.ClippingHandler(logger)
    clippingHandler.register("ranger", clipping.Ranger_thresholds)
    
    # Creating handler for importance analysis 
    # and registering commands for it
    analysisHandler = handlers.AnalysisHandler(logger)
    analysisHandler.register("l1-norm", imp.L1_norm)
    analysisHandler.register("vul-gain", imp.vulnerability_gain)
    analysisHandler.register("salience", imp.Salience)

    _, model_params, model_macs = test_func(model, testloader, device, dummy_input, logger)

    pu = utils.prune_utils(model, trainloader, classes_count, pruning_method, device)

    sorted_model = pu.channel_sorting(model, analysisHandler, importance_command)
    logger.info("channels are sorted")
    
    hr = utils.hardening_utils(hardening_ratio, clipping_command)
    hr.thresholds_extraction(sorted_model, clippingHandler, clipping_command, trainloader, device, logger)
    hardened_model = hr.relu_replacement(sorted_model)          #default: ranger. TODO: fitact, ft-clipact, proact!
    hardened_model = hr.conv_replacement(hardened_model)        #replace all Conv2d with HardenedConv2d
    logger.info(f"model is hardened: {hardened_model}")

    log_dir = logger.handlers[0].baseFilename.split("log")[0]
    torch.save(hardened_model.state_dict(), f'{log_dir}/../hardened_model-{importance_command}-{hardening_ratio}.pth')
    logger.info("model is hardened and saved")

    _, hardened_params, hardened_macs = test_func(hardened_model, testloader, device, dummy_input, logger)    

    logger.info(f"MACs overhead: {hardened_params / model_params}")
    logger.info(f"Params overhead: {hardened_macs / model_macs}")



def weights_FI_simulation(model: nn.Module, 
                          dataloader: DataLoader, 
                          repetition_count: int, 
                          BER: float, 
                          classes_count: int, 
                          device: Union[torch.device, str],
                          logger: logging.Logger):

    with torch.no_grad():
        golden_predicted, golden_accuracy = fault_simulation.golden_model_evaluation(model, dataloader, device)

        model_copy = copy.deepcopy(model)
        faulty_accuracy_total = 0
        DUE_total = 0
        SDC_critical_total = 0
        SDC_non_critical_total = 0

        #repeat FI campaign for fault_count times
        faulty_params_count_total = 0
        fault_counter = fault_simulation.fault_counter()
        for _ in range(repetition_count):
            model_copy = fault_simulation.weights_FI(model_copy, BER, device, fault_counter)
            faulty_params_count_total += fault_counter.fault_count
            
            faulty_accuracy, DUE, SDC_critical, SDC_non_critical = fault_simulation.faulty_model_evaluation(model_copy, dataloader, classes_count, golden_predicted, device)
            faulty_accuracy_total += faulty_accuracy
            DUE_total += DUE
            SDC_critical_total += SDC_critical
            SDC_non_critical_total += SDC_non_critical

            del model_copy
            model_copy = copy.deepcopy(model)
    
    logger.info(f"average number of faults: {faulty_params_count_total / repetition_count}")
    logger.info(f"golden accuracy: {golden_accuracy * 100}%")
    logger.info(f"weight FI, average accuracy: {faulty_accuracy_total * 100 / (repetition_count)}%")
    logger.info(f"average DUE: {DUE_total * 100 / repetition_count}%")
    logger.info(f"average critical SDC: {(SDC_critical_total) * 100 / repetition_count}%")
    logger.info(f"average non-critical SDC: {SDC_non_critical_total * 100 / repetition_count}")