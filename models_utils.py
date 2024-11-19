import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from typing import Union

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet18, ResNet18_Weights


def load_model(model_name: str, 
               dataset_name: str, 
               device: Union[torch.device, str] = torch.device('cuda'),
               ) -> nn.Module:
    
    if 'cifar' in dataset_name:
        full_name = dataset_name + "_" + model_name
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", full_name, pretrained=True)

    elif 'imagenet' in dataset_name:
        if model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

        elif model_name == "resnet34":
            model = resnet34(weights=ResNet34_Weights.DEFAULT).to(device)
        
        elif model_name == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)

    return model.to(device)


def load_params(model: nn.Module, 
                addr: str, 
                device: Union[torch.device, str] = torch.device('cpu'),
                ) -> None:
    i = 0
    state_dict_load = torch.load(addr, map_location=device)
    sd = model.state_dict()
    for layer_name, _ in sd.items():
        if 'num_batches_tracked' not in layer_name:           
            sd[layer_name] = nn.Parameter(state_dict_load[list(state_dict_load)[i]].to(device))
        i += 1
    
    model.load_state_dict(sd)


def train(model: nn.Module,
          dataloader: DataLoader,
          criterion: nn.Module,
          optimizer: Optimizer,
          scheduler: LambdaLR,
          callbacks = None,
          device=torch.device('cuda')) -> None:
    
    model.train()

    for data in dataloader:
        inputs, targets = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        # Forward inference
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward propagation
        loss.backward()

        # Update optimizer and LR scheduler
        optimizer.step()
        scheduler.step()

        if callbacks is not None:
            for callback in callbacks:
                callback()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device=torch.device('cuda')) -> float:
    
    model.to(device)
    model.eval()
    num_samples = 0
    num_correct = 0

    for data in dataloader:
        inputs, targets = data[0].to(device), data[1].to(device)

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

        #break

    return (num_correct / num_samples * 100).item()


