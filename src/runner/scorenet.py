from typing import Dict, Tuple

from src.data import Data
from src.runner.base import Runner
from src.loss.dsm import dsm_score_estimation
from src.loss.ssm import sliced_score_estimation_vr
from src.utils import check_input_dim
from src.sampling import langevin_sampling

import numpy as np

import torch 
from torch.utils.data.dataloader import DataLoader

from tqdm.notebook import trange


class ScoreNetRunner(Runner):
    """Runner for score-based generative model. 
    The loss function is either sliced score matching (SSM) or denoising score matching (DSM)."""

    def __init__(self, config: Dict, data: Data): 
        super().__init__(config) 

        if data.train_loader is None or data.test_loader is None:
            raise ValueError("Data loaders are not defined.")

        input_dim = iter(data.train_loader).next()[0].shape[-1]

        if not check_input_dim(self._cfg_model, input_dim):
            raise ValueError("Input dimension specified in config does not match actual input dimension.")
        
        self.train_loader, self.test_loader = data.train_loader, data.test_loader

    def __repr__(self) -> str:
        return f"ScoreNetRunner(config={self.config})"

    def train(self) -> Tuple:
        step = 0
        test_iter = iter(self.test_loader)

        train_losses, test_losses = [], []  
        loop = trange(self._cfg_training["n_epochs"])

        for epoch in loop:
            running_train_loss, running_test_loss = [], []

            for i, (X, y) in enumerate(self.train_loader):
                step += 1
                X = X.to(self.config["device"])
                
                if self._cfg_data["logit_transform"]:
                    X = self.logit_transform(X)

                score_net = lambda x: self.model(x)

                if self._cfg_training["algo"] == "ssm":
                    X = X + torch.randn_like(X) * self._cfg_training["noise_std"]
                    loss, *_ = sliced_score_estimation_vr(score_net, X.detach(), n_particles=1)

                elif self._cfg_training["algo"] == "dsm":
                    loss = dsm_score_estimation(score_net, X, sigma=self._cfg_training["noise_std"])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_train_loss.append(loss.item()) 

                if step >= self._cfg_training["n_iters"]:
                    return train_losses, test_losses

                if step % self._cfg_training["eval_freq"] == 0:
                    try:
                        test_X, _ = next(test_iter)
                    except StopIteration:
                        test_iter = iter(self.test_loader)
                        test_X, _ = next(test_iter)

                    test_X = test_X.to(self.config["device"])

                    if self._cfg_data["logit_transform"]:
                        test_X = self.logit_transform(test_X)

                    if self._cfg_training["algo"] == "ssm":
                        test_X += torch.randn_like(test_X) * self._cfg_training["noise_std"]
                        test_loss, *_ = sliced_score_estimation_vr(score_net, test_X.detach(), n_particles=self._cfg_training["n_particles"])

                    elif self._cfg_training["algo"] == "dsm":
                        test_loss = dsm_score_estimation(score_net, test_X, sigma=self._cfg_training["noise_std"])

                    running_test_loss.append(test_loss.item())

                if step % self._cfg_training["snapshot_freq"] == 0:
                    self.save_states(step) 

                if step == self._cfg_training["n_iters"]:
                    return train_losses, test_losses
                
            if self.scheduler is not None:
                self.scheduler.step()
            
            if step % self._cfg_training["eval_freq"] == 0:
                avg_train_loss = np.mean(running_train_loss)
                train_losses.append(avg_train_loss)  

                avg_test_loss = np.mean(running_test_loss)
                test_losses.append(avg_test_loss)     

                msg = f"Epoch: {epoch+1} | Train loss: {avg_train_loss:.5f} | Val loss: {avg_test_loss:.5f}"
                loop.set_description(msg)


        return train_losses, test_losses

    def generate(self) -> torch.Tensor:
        """Generate samples from the model.
        
        Args:
            n_batches (int): Number of n_batches to generate. Defaults to 1.
        
        Returns:
            torch.Tensor: Generated samples."""
        self.model.eval()

        n_batches = self._cfg_sampling["langevin"]["n_batches"]
        n_steps = self._cfg_sampling["langevin"]["n_steps"]
        step_lr = self._cfg_sampling["langevin"]["step_lr"]

        reference_samples, _ = iter(self.test_loader).next()
        init_samples = torch.rand_like(reference_samples)
        synthetic_samples = []

        loop = trange(n_batches, desc=f"Generating {n_batches} batches with {init_samples.shape[0]} samples each.")

        for _ in loop:
            all_samples = langevin_sampling(init_samples, self.model, n_steps, step_lr) 
            new_samples = all_samples[-1]

            if self._cfg_data["logit_transform"]:
                new_samples = torch.sigmoid(new_samples)

            synthetic_samples.append(new_samples)

        synthetic_samples = torch.cat(synthetic_samples, dim=0)
        return synthetic_samples.detach().cpu()