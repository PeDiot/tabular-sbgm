from typing import List

import torch
import torch.nn as nn

import numpy as np
from tqdm.notebook import trange


def langevin_dynamics(x: torch.Tensor, scorenet: nn.Module, n_steps: int=1000, step_lr: float=0.00002) -> List:
    """Apply Langevin dynamics  to sample from a distribution using only its score function.
    
    Args:
        x (torch.Tensor): Initial sample.
        scorenet (nn.Module): Score network.
        n_steps (int, optional): Number of steps to run the dynamics. Defaults to 1000.
        step_lr (float, optional): Step size. Defaults to 0.00002.
    
    Returns:
        List: List of MCMC samples."""

    new_x = []

    with torch.no_grad():
        for _ in range(n_steps):
            new_x.append(torch.clamp(x, 0.0, 1.0).to("cpu"))
            noise = torch.randn_like(x) * np.sqrt(step_lr * 2)
            grad = scorenet(x)
            x = x + step_lr * grad + noise

        return new_x
    
def anneal_langevin_dynamics(
    x: torch.Tensor, 
    scorenet: nn.Module, 
    sigmas: torch.Tensor, 
    n_steps_each: int=100, 
    step_lr: float=0.00002) -> List:
    """Apply annealed Langevin dynamics to sample from a distribution using only its score function.
    
    Args:
        x (torch.Tensor): Initial sample.
        scorenet (nn.Module): Score network.
        sigmas (torch.Tensor): List of sigmas to anneal over.
        n_steps_each (int, optional): Number of steps to run the dynamics for each sigma. Defaults to 100.
        step_lr (float, optional): Step size. Defaults to 0.00002.
    
    Returns:
        List: List of MCMC samples."""
    
    new_x = []

    loop = trange(len(sigmas), desc=f"Annealing Langevin dynamics over {len(sigmas)} sigmas.") 
    
    with torch.no_grad():
        for c in loop:
        
            sigma = sigmas[c]
            # labels = torch.ones(x.shape[0], device=x.device) * c
            # labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            
            for s in range(n_steps_each):
                new_x.append(torch.clamp(x, 0.0, 1.0).to("cpu"))
                noise = torch.randn_like(x) * np.sqrt(step_size * 2)
                grad = scorenet(x)                                      # scorenet(x, labels)
                x = x + step_size * grad + noise
                
    return new_x