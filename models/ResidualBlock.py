import torch
from torch import nn

from models.RMSNorm import RMSNorm
from monitor.TrackableModule import TrackableModule


class ResidualBlock(TrackableModule):
    """Residual block with pre-normalization and Mamba module.

    Applies RMS normalization followed by a Mamba block, with residual
    connection around the normalized block.

    Attributes:
      d_model : int
        Data dimension.
      mamba_block : nn.Module
        Mamba block module.
      rmsnorm : RMSNorm
        Root mean square normalization layer.
    """

    def __init__(
        self, d_model: int, mamba_block: nn.Module, eps: float, device: torch.device
    ):
        super().__init__()
        self.d_model = d_model
        self.mamba_block = mamba_block

        self.rmsnorm = RMSNorm(d_model, eps=eps, device=device)

    def forward(self, u):
        """Apply normalization, Mamba block, and residual connection.

        Parameters:
          u : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
          torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        y = u + self.mamba_block(self.rmsnorm(u))

        return y


# Number of parameters per object
# total numer of paramters = rmsnorm + mamba_block = d_model + 3*d_inner*d_model (+ d_inner if bias = True) (+ d_model if bias= True) + 3*d_state*d_inner + d_inner*d_conv (+ d_inner if bias true)
