"""Transformer final layer.

Applies layer normalization and a small readout network to the final hidden
state produced by a transformer backbone.
"""

import torch
import torch.nn as nn

from monitor.TrackableModule import TrackableModule


class TransformerFinalLayer(TrackableModule):
    """Builds the final prediction head for transformer outputs.

    The layer normalizes the last hidden state and maps it to the requested
    output dimension through a two-layer MLP.
    """

    def __init__(self, cfg, device):
        """Initialize the final layer.

        Parameters:
          cfg: Configuration object containing model dimensions and activation
            settings.
          device: Torch device on which the module parameters should live.
        """
        super().__init__()

        self.device = device
        self.model_dim = cfg.model.model_dim
        self.output_dim = cfg.model.output_dim
        self.activation_function = cfg.model.activation_function

        self.norm = nn.LayerNorm(self.model_dim, device=self.device, dtype=torch.float)
        self.readout = nn.Sequential(
            nn.Linear(
                self.model_dim,
                max(16, self.model_dim // 2),
                device=self.device,
                dtype=torch.float,
            ),
            self._activation(self.activation_function),
            nn.Linear(
                max(16, self.model_dim // 2),
                self.output_dim,
                device=self.device,
                dtype=torch.float,
            ),
        )

    def _activation(self, name):
        """Return the requested activation layer.

        Parameters:
          name: Name of the activation function to use.

        Returns:
          torch.nn.Module: The activation module matching the requested name.

        Raises:
          ValueError: If the activation name is not supported.
        """
        if name.lower() == "gelu":
            return nn.GELU()
        elif name.lower() == "tanh":
            return nn.Tanh()
        elif name.lower() == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, last_hidden):
        """Apply normalization and the readout network to hidden states.

        Parameters:
          last_hidden: Tensor of shape (batch, seq_len, features).

        Returns:
          torch.Tensor: Output predictions for the provided hidden states.

        Raises:
          ValueError: If the input tensor is not three-dimensional.
        """
        if last_hidden.dim() != 3:
            raise ValueError(
                "Expected input of shape (batch, seq_len, features); "
                f"got {tuple(last_hidden.shape)}"
            )

        last_hidden = self.norm(last_hidden)
        return self.readout(last_hidden)
