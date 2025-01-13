import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import models_utils
import torchprofile
import importance_analysis as imp
import copy
import utils
import pruning
from typing import Union, Dict
import logging
import handlers
import clipping
import fault_simulation
import time

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
    handler.register("deepvigor", imp.DeepVigor)

    #model_cp = copy.deepcopy(model)
    pu = utils.prune_utils(model, trainloader, classes_count, pruning_method, device)
    pu.set_pruning_ratios(pruning_ratio)

    sorted_model = pu.channel_sorting(model, handler, logger, importance_command, pruning_ratio, 0)     #hardening ratio is assumed to be 0
    
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
                   pruning_ratio: float,
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
    analysisHandler.register("deepvigor", imp.DeepVigor)

    _, model_params, model_macs = test_func(model, testloader, device, dummy_input, logger)

    pu = utils.prune_utils(model, trainloader, classes_count, pruning_method, device)

    sorted_model = pu.channel_sorting(model, analysisHandler, logger, importance_command, pruning_ratio, hardening_ratio)
    
    hr = utils.hardening_utils(hardening_ratio, clipping_command)
    hr.thresholds_extraction(sorted_model, clippingHandler, clipping_command, trainloader, device, logger)
    hardened_model = hr.relu_replacement(sorted_model)          #default: ranger. TODO: fitact, ft-clipact, proact!
    hardened_model = hr.conv_replacement(hardened_model)        #replace all Conv2d with HardenedConv2d
    logger.info(f"model is hardened: {hardened_model}")

    log_dir = logger.handlers[0].baseFilename.split("log")[0]
    torch.save(hardened_model.state_dict(), f'{log_dir}/../hardened_model-{importance_command}-{pruning_ratio}-{hardening_ratio}.pth')
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
                          logger: logging.Logger) -> None:

    with torch.no_grad():
        golden_predicted, golden_accuracy = fault_simulation.model_evaluation(model, dataloader, device)

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


def channel_ranking_func(model: nn.Module,
                         dataloader: DataLoader,
                         command: str,
                         classes_count: int,
                         logger: logging.Logger,
                         device: Union[torch.device, str]) -> None:
    
    handler = handlers.AnalysisHandler(logger)
        
    # registering commands for importance analysis 
    handler.register("l1-norm", imp.L1_norm)
    handler.register("vul-gain", imp.vulnerability_gain)
    handler.register("salience", imp.Salience)
    handler.register("deepvigor", imp.DeepVigor)
    handler.register("channel-FI", imp.channel_FI)

    if command == "l1-norm":
        sort_index_conv_dict = handler.execute(command, model, ...)  
        
    elif command == "vul-gain":
        sort_index_conv_dict = handler.execute(command, model, dataloader, classes_count, device)

    elif command == "salience":
        sort_index_conv_dict = handler.execute(command, model, classes_count, device)

    elif command == "deepvigor":
        for data in dataloader:
            inputs = data[0].to(device)
            break
        sort_index_conv_dict = handler.execute(command, model, inputs, classes_count, device, logger)

    elif command == "channel-FI":
        sort_index_conv_dict = handler.execute(command, model, dataloader, device, logger)

    else:
        raise Exception(f"Unexpected analysis command is given: {command}")

    assert len(sort_index_conv_dict) != 0
    save_dict(sort_index_conv_dict, command, logger)
    logger.info(f"channels are sorted and svaed based on {command}")


def save_dict(sort_index_dict: Dict,
              command: str,
              logger: logging.Logger) -> None:
    for key in sort_index_dict:
        log_dir = logger.handlers[0].baseFilename.split("log")[0]
        file = open(f"{log_dir}/channel_ranking-{command}-{key}.txt", 'w')
        for i in sort_index_dict[key]:
            file.write(str(i.item()) + "\n")
        file.close()


def performance_func(model: nn.Module,
                    dummy_input: torch.tensor,
                    logger: logging.Logger) -> Union[float, int, int]:
    

    warmup_count = 100
    for _ in range(warmup_count):
        _ = model(dummy_input)
    
    eval_count = 10000

    total_time = 0
    for _ in range(eval_count):
        tmp_time1 = time.time()
        _ = model(dummy_input)
        tmp_time2 = time.time()
        total_time += tmp_time2 - tmp_time1
    performance = total_time / eval_count

    logger.info(f"model average performance with {eval_count} repetition: {performance * 1000} ms")
    
