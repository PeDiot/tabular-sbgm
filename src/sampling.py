from typing import List

import torch
import torch.nn as nn

import numpy as np


def langevin_sampling(x: torch.Tensor, scorenet: nn.Module, n_steps: int=1000, step_lr: float=0.00002) -> List:
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