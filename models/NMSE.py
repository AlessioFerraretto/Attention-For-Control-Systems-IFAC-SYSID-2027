# Normalized Mean Squared Error (NMSE) or Relative Squared Error (RSE) loss function for regression tasks.
# This loss function is particularly useful when the scale of the target variable varies significantly across samples.
import torch
from torch import nn

from utils.Globals import EPS


class NMSE(nn.Module):
    """Computes the NMSE: ||y - y_hat||^2 / (||y||^2 + eps)."""

    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Sum the squared differences along the sequence dimension
        squared_error = torch.sum((preds - targets) ** 2, dim=-1)

        # Sum the squared targets along the sequence dimension
        target_energy = torch.sum(targets**2, dim=-1)

        # Divide the error by the energy of the target for each sample in the batch,
        # then return the mean across the batch.
        return torch.mean(squared_error / (target_energy + EPS))
