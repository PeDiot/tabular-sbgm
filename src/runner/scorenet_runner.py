from typing import Dict, Optional

from src.data import Data
from src.runner.base import Runner
from src.loss.dsm import dsm_score_estimation
from src.loss.ssm import sliced_score_estimation
from src.utils import check_input_dim, save_pickle
from src.sampling import langevin_dynamics

import numpy as np
import torch 
from tqdm.notebook import trange


class ScoreNetRunner(Runner):
    """Runner for score-based generative model without noise perturbations.

    Details: The loss function is either sliced score matching (SSM) or score matching (DSM)."""

    def __init__(self, config: Dict, data: Data): 
        super().__init__(config) 

        if data.train_loader is None or data.test_loader is None:
            raise ValueError("Data loaders are not defined.")

        self._cfg_model = check_input_dim(self._cfg_model, data.input_dim) 
                
        self.train_loader, self.test_loader = data.train_loader, data.test_loader

    def __repr__(self) -> str:
        return f"ScoreNetRunner(config={self.config})"

    def train(self) -> Dict:
        """Train and evaluate score-based model. Returns a dictionary with training and test losses."""

        step = 0
        n_steps_no_improvement = 0
        prev_avg_test_loss = np.inf

        test_iter = iter(self.test_loader)

        loss_tracker = {"train": [], "test": []}
        loop = trange(self._cfg_training["n_epochs"])

        for epoch in loop:
            running_train_loss, running_test_loss = [], []

            for X, _ in self.train_loader:
                step += 1
                X = X.to(self.config["device"])
                
                if self._cfg_data["logit_transform"]:
                    X = self.logit_transform(X)

                score_net = lambda x: self.model(x)

                if self._cfg_training["algo"] == "ssm":
                    X = X + torch.randn_like(X) * self._cfg_training["noise_std"]
                    loss, *_ = sliced_score_estimation(score_net, X.detach(), n_particles=self._cfg_training["n_particles"])

                elif self._cfg_training["algo"] == "dsm":
                    loss = dsm_score_estimation(score_net, X, sigma=self._cfg_training["noise_std"])
                
                else:
                    raise ValueError(f"Invalid algorithm: {self._cfg_training['algo']}")
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_train_loss.append(loss.item()) 

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
                        test_loss, *_ = sliced_score_estimation(score_net, test_X.detach(), n_particles=self._cfg_training["n_particles"])

                    elif self._cfg_training["algo"] == "dsm":
                        test_loss = dsm_score_estimation(score_net, test_X, sigma=self._cfg_training["noise_std"])

                    running_test_loss.append(test_loss.item())

                    avg_train_loss = np.mean(running_train_loss)
                    avg_test_loss = np.mean(running_test_loss)

                    loss_tracker["train"].append(avg_train_loss) 
                    loss_tracker["test"].append(avg_test_loss)

                    test_loss_diff = np.abs(avg_test_loss - prev_avg_test_loss)

                    if step > self._cfg_training["n_steps_min"] and test_loss_diff < self._cfg_training["stop_threshold"]:
                        n_steps_no_improvement += 1
                
                        if n_steps_no_improvement >= self._cfg_training["n_steps_no_improvement"]:

                            if self._cfg_backup["save"]: 
                                self.save_states(step) 
                                save_pickle(loss_tracker, f"{self._cfg_backup['dir']}/loss.pkl")

                            return loss_tracker                 

                    msg = f"Epoch: {epoch+1} | Train loss: {avg_train_loss:.5f} | Val loss: {avg_test_loss:.5f}"
                    loop.set_description(msg)

                    prev_avg_test_loss = avg_test_loss

                if step % self._cfg_training["snapshot_freq"] == 0 and self._cfg_backup["save"]:
                    self.save_states(step) 
                
            if self.scheduler is not None:
                self.scheduler.step()

        if self._cfg_backup["save"]:
            self.save_states(step) 
            save_pickle(loss_tracker, f"{self._cfg_backup['dir']}/loss.pkl")

        return loss_tracker

    def sample(self, n_batches: Optional[int]=None) -> torch.Tensor:
        """Generate samples from the model.
        
        Args:
            n_batches (int): Number of n_batches to generate. Defaults to 1.
        
        Returns:
            torch.Tensor: Generated samples."""
        self.model.eval()

        if n_batches is None:
            n_batches = self._cfg_sampling["n_batches"]

        n_steps = self._cfg_sampling["n_steps"]
        step_lr = self._cfg_sampling["step_lr"]

        reference_samples, _ = iter(self.test_loader).next()
        init_samples = torch.rand_like(reference_samples)
        synthetic_samples = []

        loop = trange(n_batches, desc=f"Generating {n_batches} batches with {init_samples.shape[0]} samples each using Langevin dynamics.")

        for _ in loop:
            all_samples = langevin_dynamics(init_samples, self.model, n_steps, step_lr) 
            n_samples = len(all_samples)

            new_samples = torch.stack(all_samples[int(n_samples * self._cfg_sampling["burn_in"]):], dim=0)
            new_samples = torch.mean(new_samples, dim=0)

            if self._cfg_data["logit_transform"]:
                new_samples = torch.sigmoid(new_samples)

            synthetic_samples.append(new_samples)

        synthetic_samples = torch.cat(synthetic_samples, dim=0)
        return synthetic_samples.detach().cpu()