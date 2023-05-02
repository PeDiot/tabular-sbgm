import torch
import torch.nn as nn
import torch.autograd as autograd


def dsm(energy_net: nn.Module, samples: torch.Tensor, sigma: int=1) -> torch.Tensor:
    """Compute denoising score matching for a given energy network, samples and sigma pertubation."""

    samples.requires_grad_(True)

    vector = torch.randn_like(samples) * sigma
    kernel = vector
    perturbed_inputs = samples + vector

    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.
    return loss

def dsm_score_estimation(scorenet: nn.Module, samples: torch.Tensor, sigma: float=0.01) -> torch.Tensor:
    """Compute the estimation of score matching divergence for a given score network, samples and sigma pertubation."""

    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)

    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)

    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
    return loss

def anneal_dsm_score_estimation(
    scorenet: nn.Module, 
    samples: torch.Tensor, 
    labels: torch.Tensor, 
    sigmas: torch.Tensor, 
    anneal_power: float=2.
) -> torch.Tensor:
    """Compute the estimation of annealed score matching divergence by applying multiple sigma pertubations."""
    
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas

    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)

    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)

    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0)