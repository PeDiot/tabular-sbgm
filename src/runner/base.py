from typing import Dict
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from src.model import MLPScoreNet
from src.utils import save_config


class Runner():
    def __init__(self, config: Dict):
        self.config = config

        self._cfg_data = self.config["data"]
        self._cfg_model = self.config["model"]
        self._cfg_training = self.config["training"]
        self._cfg_optim = self.config["optim"]
        self._cfg_sampling = self.config["sampling"]
        self._cfg_backup = self.config["backup"]
        
        self.model = MLPScoreNet(self._cfg_model).to(self.config["device"])
        self.optimizer = self.get_optimizer(self.model.parameters())

        if self._cfg_optim["use_scheduler"]:
            self.scheduler = self.get_scheduler()
        else:
            self.scheduler = None
        
        if self._cfg_training["resume_training"]:
            self.load_states()

        self._save_model_cfg = False

    def get_optimizer(self, parameters: torch.Tensor) -> optim.Optimizer:
        cfg_optimizer = self._cfg_optim["optimizer"]

        if cfg_optimizer["name"] == "Adam":
            return optim.Adam(
                parameters, 
                lr=cfg_optimizer["lr"], 
                weight_decay=cfg_optimizer["weight_decay"], 
                betas=(cfg_optimizer["beta1"], 0.999)
            )
        
        elif cfg_optimizer["name"] == "RMSProp":
            return optim.RMSprop(parameters, lr=cfg_optimizer["lr"], weight_decay=cfg_optimizer["weight_decay"])
        
        elif cfg_optimizer["name"] == "SGD":
            return optim.SGD(parameters, lr=cfg_optimizer["lr"], momentum=0.9)
        
        else:
            raise NotImplementedError(f"Optimizer {cfg_optimizer['optimizer']} not understood.")

    def get_scheduler(self) -> _LRScheduler:
        cfg_scheduler = self._cfg_optim["scheduler"]

        if cfg_scheduler["name"] == "StepLR":
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=cfg_scheduler["step_size"], 
                gamma=cfg_scheduler["gamma"]
            )
        
        elif cfg_scheduler["name"] == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg_scheduler["gamma"])
        
        elif cfg_scheduler["name"] == "MultiplicativeLR": 
            func = lambda epoch: cfg_scheduler["lambda"]
            return optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=func)
        
        else:
            raise NotImplementedError(f"Scheduler {cfg_scheduler['name']} not understood.")
        
    def logit_transform(self, x: torch.Tensor, lam: float=1e-6):
        x = lam + (1 - 2 * lam) * x
        x = torch.log(x) - torch.log1p(-x)

        return x
    
    def save_states(self, step: int): 

        states = [
            self.model.state_dict(),
            self.optimizer.state_dict()
        ]

        if self.scheduler is not None:
            states.append(self.scheduler.state_dict())

        backup_path = self._cfg_backup["dir"]

        if not os.path.exists(backup_path):
            os.makedirs(backup_path)

        torch.save(states, f"{backup_path}/checkpoint_{step}.pth")
        torch.save(states, f"{backup_path}/checkpoint.pth")

        msg = f"Saved checkpoint at {backup_path}/checkpoint_{step}.pth."

        if not self._save_model_cfg: 
            save_config(self._cfg_model, f"{backup_path}/model_cfg.yml")
            self._save_model_cfg = True
            msg += f" Saved model config at {backup_path}/model_cfg.yml."

        print(msg)

    def load_states(self): 
        if self._cfg_backup["checkpoint"] == -1: 
            path = f"{self._cfg_backup['dir']}/checkpoint.pth"
        else:
            path = f"{self._cfg_backup['dir']}/checkpoint_{self._cfg_backup['checkpoint']}.pth"
            
        states = torch.load(path)

        if len(states) == 2:
            model_states, optimizer_states = states

        elif len(states) == 3 and self._cfg_optim["use_scheduler"]: 
            model_states, optimizer_states, scheduler_states = states
            self.scheduler.load_state_dict(scheduler_states)

        else:
            raise ValueError("Invalid number of states in checkpoint.")
        
        self.model.load_state_dict(model_states)
        self.optimizer.load_state_dict(optimizer_states)