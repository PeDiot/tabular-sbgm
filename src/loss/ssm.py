from typing import Tuple

import torch
import torch.nn as nn
import torch.autograd as autograd

def sliced_score_estimation(score_net, samples, n_particles=1) -> Tuple:
    """Sliced score matching estimation.
    
    Args:
        score_net (nn.Module): Score network.
        samples (torch.Tensor): Samples from the data distribution.
        n_particles (int, optional): Number of particles. Defaults to 1.
    
    Returns:
        Tuple: ssm loss, gradient loss, hessian loss."""

    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * vectors, dim=-1) ** 2 * 0.5
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()

def sliced_score_estimation_vr(score_net: nn.Module, samples: torch.Tensor, n_particles: int=1) -> Tuple:
    """Sliced score matching estimation with variance reduction.
    
    Args:
        score_net (nn.Module): Score network.
        samples (torch.Tensor): Samples from the data distribution.
        n_particles (int, optional): Number of particles. Defaults to 1.
    
    Returns:
        Tuple: ssm loss, gradient loss, hessian loss."""
    
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]

    grad1 = grad1.view(dup_samples.shape[0], -1)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.

    loss2 = torch.sum((vectors * grad2).view(dup_samples.shape[0], -1), dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()