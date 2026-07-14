import torch
from torch import nn

from monitor.TrackableModule import TrackableModule


class RMSNorm(TrackableModule):
    """Root mean square normalization layer.

    Implements RMSNorm as a learnable normalization technique with
    per-feature scale parameters.

    Attributes:
      eps : float
        Small constant to avoid division by zero.
      weight : torch.nn.Parameter
        Learnable scale parameters with shape (d_model,).
    """

    # TODO: Remove default
    def __init__(self, d_model: int, eps: float, device: torch.device):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        """Apply RMSNorm normalization.

        Parameters:
          x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
          torch.Tensor: Normalized tensor of shape (batch_size, sequence_length, d_model).
        """
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )

        return output
