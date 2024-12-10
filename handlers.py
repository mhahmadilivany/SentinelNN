import torch
import torch.nn as nn
from typing import Callable, Dict, Any
from torch.utils.data import DataLoader
from typing import Union
import os
import logging
from datetime import datetime


class LogHandler():
    def __init__(self,
                 run_mode: str,
                 model_name: str,
                 dataset_name: str,
                 ) -> None:
        self.run_mode = run_mode
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.log_direction = model_name + "-" + dataset_name + "/" + run_mode


    def getLogger(self) -> logging.Logger:
        if not os.path.exists(self.log_direction):
            os.makedirs(self.log_direction)
        log_file_name = f"{self.log_direction}/log-{self.run_mode}-{self.model_name}-{self.dataset_name}-{datetime.now().strftime('%Y-%m-%d %H:%M')}.txt"
        logging.basicConfig(
            filename=log_file_name,
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s : %(message)s',
        )
        logger = logging.getLogger()
        logger.info(f"run-mode: {self.run_mode}")
        return logger
    

class AnalysisHandler():
    def __init__(self,
                 logger: logging.Logger) -> None:
        self.logger = logger
        self.functions: Dict[str, Callable] = {}

    def register(self,
                 command: str,
                 func: Callable) -> None:
        self.functions[command] = func

    def execute(self, 
                command: str, 
                *args, **kwargs) -> Any:
        if command in self.functions:
            self.logger.info(f"Executing '{command}'")
            obj = self.functions[command](*args, **kwargs)
            return obj(*args, **kwargs)
        else:
            self.logger.error(f"Unknown command: {command}. Available commands: {list(self.functions.keys())}")
            raise ValueError(f"Unknown command: {command}. Available commands: {list(self.functions.keys())}")


class RunModeHandler():
    def __init__(self, 
                 logger: logging.Logger) -> None:
        self.logger = logger
        self.run_modes: Dict[str, Callable] = {}
    
    def register(self,
                 run_mode: str,
                 func: Callable) -> None:
        self.run_modes[run_mode] = func

    def execute(self,
                run_mode: str, 
                *args, **kwargs):
        if run_mode in self.run_modes:
            self.logger.info(f"Executing '{run_mode}'")
            func = self.run_modes[run_mode]  #(*args, **kwargs)
            return func(*args, **kwargs)
        else:
            self.logger.error(f"Unknown run-mode: {run_mode}. Available run-modes: {list(self.run_modes.keys())}")
            raise ValueError(f"Unknown run-mode: {run_mode}. Available run-modes: {list(self.run_modes.keys())}")

