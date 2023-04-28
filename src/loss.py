import torch
import torch.nn as nn
import torch.autograd as autograd

from typing import Union


def dsm(energy_net: nn.Module, samples: torch.Tensor, sigma: int=1) -> torch.Tensor:
    """Compute the score matching divergence for a given energy network, samples and sigma pertubation."""

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

class ScoreMatchinLoss(nn.Module):
    """Custom loss function based on score-matching."""

    def __init__(self, network: nn.Module, sigma: Union[float, int]=1, score_estimation: bool=False):
        super().__init__()

        self.network = network
        self.sigma = sigma

        if score_estimation:
            self.func = dsm_score_estimation
        else:
            self.func = dsm

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.func(self.network, samples, self.sigma)
    