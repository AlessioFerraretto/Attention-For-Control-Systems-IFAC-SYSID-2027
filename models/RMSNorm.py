import torch
from torch import nn

from monitor.TrackableModule import TrackableModule
from utils.Globals import EPS


class RMSNorm(TrackableModule):
    """Root mean square normalization layer.

    Implements RMSNorm as a learnable normalization technique with
    per-feature scale parameters.

    Attributes:
        Small constant to avoid division by zero.
      weight : torch.nn.Parameter
        Learnable scale parameters with shape (d_model,).
    """

    # TODO: Remove default
    def __init__(self, d_model: int, device: torch.device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        """Apply RMSNorm normalization.

        Parameters:
          x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
          torch.Tensor: Normalized tensor of shape (batch_size, sequence_length, d_model).
        """
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * self.weight

        return output
