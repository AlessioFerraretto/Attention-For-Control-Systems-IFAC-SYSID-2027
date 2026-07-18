"""Positional encoding for transformer inputs.

This module projects input features to the model dimension and adds learned
sinusoidal positional information before the transformer stack.
"""

import torch
from torch import nn

from monitor.TrackableModule import TrackableModule


class TransformerPositionalEncoding(TrackableModule):
    """Project input features and add positional encodings.

    The module maps the input feature dimension to the transformer hidden size
    and combines it with a positional signal used by the attention layers.
    """

    def __init__(
        self,
        cfg,
        device,
    ):
        """Initialize the positional encoding module.

        Parameters:
          cfg: Configuration object containing input and model dimensions.
          device: Torch device on which the module parameters should live.
        """
        super().__init__()

        self.device = device
        self.input_dim = cfg.model.input_dim
        self.model_dim = cfg.model.model_dim

        self.input_proj = nn.Linear(
            self.input_dim, self.model_dim, device=device, dtype=torch.float
        )

        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1, self.model_dim, device=self.device)
        )

    def _positional_embeddings(self, x):
        """Create positional embeddings for a batch of sequences.

        Parameters:
          x: Input tensor of shape (batch, seq_len, features).

        Returns:
          torch.Tensor: Positional embeddings of shape (1, seq_len, features).
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        div_term = torch.exp(
            torch.arange(0, self.model_dim, 2, device=x.device, dtype=x.dtype)
            * (-torch.log(torch.tensor(10000.0, device=x.device)) / self.model_dim)
        )
        pe = torch.zeros(seq_len, self.model_dim, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term.unsqueeze(0))
        if self.model_dim > 1:
            pe[:, 1::2] = torch.cos(
                positions.unsqueeze(1) * div_term.unsqueeze(0)[: pe[:, 1::2].shape[1]]
            )
        return pe.unsqueeze(0).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the input and add positional information.

        Parameters:
          x: Input tensor of shape (batch, seq_len, features).

        Returns:
          torch.Tensor: Hidden states with positional encodings added.

        Raises:
          ValueError: If the input tensor is not three-dimensional.
        """
        if x.dim() != 3:
            raise ValueError(
                "Expected input of shape (batch, seq_len, features); "
                f"got {tuple(x.shape)}"
            )

        projected = self.input_proj(x)
        positional_encoding = self.positional_encoding.to(device=projected.device)
        positions = self._positional_embeddings(projected) + positional_encoding
        hidden = projected + positions

        return hidden
