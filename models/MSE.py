# Mean Squared Error (MSE) loss function for regression tasks.
import torch
import torch.nn as nn


class MSE(nn.Module):
    """Computes the MSE: ||y - y_hat||^2"""

    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute the squared differences between predictions and targets
        sq_diff = (preds - targets) ** 2

        # Return the mean over all elements
        return torch.mean(sq_diff)
