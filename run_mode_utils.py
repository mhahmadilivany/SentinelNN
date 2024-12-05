import torch
import torch.nn as nn
import models_utils
import torchprofile

def test(model, testloader, device, dummy_input, logger):
    net_accuracy = models_utils.evaluate(model, testloader, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    total_macs = torchprofile.profile_macs(model, dummy_input)
    logger.info(f"model test top-1 accuracy: {net_accuracy}%")
    logger.info(f"total number of MACs: {total_macs}")
    logger.info(f"total number of parameters: {total_params}")