from typing import Dict, Optional

from src.data import Data
from src.runner.base import Runner
from src.loss.dsm import anneal_dsm_score_estimation
from src.loss.ssm import anneal_sliced_score_estimation_vr
from src.utils import check_input_dim, save_pickle
from src.sampling import anneal_langevin_dynamics

import numpy as np
import torch 
from tqdm.notebook import trange


class AnnealRunner(Runner):
    """Runner for score-based generative model with multiple noise perturbations.

    Details: The loss function is either annealed sliced score matching (SSM) or annealed score matching (DSM)."""

    def __init__(self, config: Dict, data: Data): 
        super().__init__(config) 

        if data.train_loader is None or data.test_loader is None:
            raise ValueError("Data loaders are not defined.")

        input_dim = iter(data.train_loader).next()[0].shape[-1]

        self._cfg_model = check_input_dim(self._cfg_model, input_dim) 
                
        self.train_loader, self.test_loader = data.train_loader, data.test_loader

        self.sigmas = torch.tensor(
            np.exp(
                np.linspace(
                    np.log(self._cfg_training["sigma_begin"]), 
                    np.log(self._cfg_training["sigma_end"]),
                    self._cfg_training["n_sigmas"]
                )
            )
        ).float().to(self.config["device"])

    def __repr__(self) -> str:
        return f"ScoreNetRunner(config={self.config})"

    def train(self) -> Dict:
        """Train and evaluate score-based model. 
        Returns a dictionary with training and test losses."""

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
                self.model.train()
                X = X.to(self.config["device"])
                
                if self._cfg_data["logit_transform"]:
                    X = self.logit_transform(X)

                labels = torch.randint(0, len(self.sigmas), (X.shape[0],), device=X.device)

                if self._cfg_training["algo"] == "dsm":
                    loss = anneal_dsm_score_estimation(self.model, X, labels, self.sigmas, self._cfg_training["anneal_power"])
                
                elif self._cfg_training["algo"] == "ssm":
                    loss = anneal_sliced_score_estimation_vr(
                        self.model, 
                        X, 
                        labels, 
                        self.sigmas,
                        n_particles=self._cfg_training["n_particles"])

                else:
                    raise ValueError(f"Invalid algorithm: {self._cfg_training['algo']}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_train_loss.append(loss.item()) 

                if step % self._cfg_training["eval_freq"] == 0:
                    self.model.eval()

                    try:
                        test_X, _ = next(test_iter)
                    except StopIteration:
                        test_iter = iter(self.test_loader)
                        test_X, _ = next(test_iter)

                    test_X = test_X.to(self.config["device"])

                    test_labels = torch.randint(0, len(self.sigmas), (test_X.shape[0],), device=test_X.device)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(
                            self.model,
                            test_X, 
                            test_labels, 
                            self.sigmas,
                            self._cfg_training["anneal_power"])

                    running_test_loss.append(test_dsm_loss.item())

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

        n_steps_each = self._cfg_sampling["n_steps_each"]
        step_lr = self._cfg_sampling["step_lr"]

        reference_samples, _ = iter(self.test_loader).next()
        init_samples = torch.rand_like(reference_samples)

        synthetic_samples = []

        for _ in range(n_batches):
            all_samples = anneal_langevin_dynamics(init_samples, self.model, self.sigmas, n_steps_each, step_lr) 
            new_samples = all_samples[-1]

            if self._cfg_data["logit_transform"]:
                new_samples = torch.sigmoid(new_samples)

            synthetic_samples.append(new_samples)

        synthetic_samples = torch.cat(synthetic_samples, dim=0).detach().cpu()

        return synthetic_samples