from collections.abc import Callable

import torch

from models.MSE import MSE
from utils.Globals import EPS


class FuturePrediction:
    def __init__(self, cfg):
        self.requires_y_in_input = True

    def loss_fn(self) -> Callable:
        return MSE()

    def compute_metrics(self, predictions, targets, info=None):
        """
        Computes the FIT metric by projecting the predicted kernel parameters
        into the impulse response space and comparing it with the true 'g'.
        """
        predictions = predictions.float()
        targets = targets.float()
        norm_diff = torch.linalg.vector_norm(targets - predictions, dim=-1)
        norm_true = torch.linalg.vector_norm(targets, dim=-1)
        fit_per_sample = 100.0 * (1.0 - (norm_diff / (norm_true + EPS)))
        return {"fit": torch.mean(fit_per_sample).item()}

    def compute_y_hat(self, predictions, info, u, y_with_grad):
        """Computes the estimated output y_hat using the predicted kernel parameters and the input u."""
        B, n = y_with_grad.shape
        y_hat = predictions[:, :n, -1] if predictions.dim() == 3 else predictions[:, :n]
        return y_hat
