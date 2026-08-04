"""Transformer encoder layer.

Implements a single transformer encoder block with multi-head self-attention,
residual connections, layer normalization, and a feed-forward network.
"""

import torch
import torch.nn as nn

from monitor.TrackableModule import TrackableModule


class TransformerEncoderLayer(TrackableModule):
    """Build a single encoder block for a transformer model.

    The layer composes self-attention and a position-wise feed-forward network
    with residual connections and normalization.
    """

    def __init__(self, cfg, device):
        """Initialize the encoder layer.

        Parameters:
          cfg: Configuration object with model dimensions, attention heads,
            dropout, and activation settings.
          device: Torch device used for the module parameters.
        """
        super().__init__()

        self.device = device
        self.model_dim = cfg.model.model_dim
        self.num_heads = cfg.model.num_heads
        self.activation_function = cfg.model.activation_function
        self.dropout = cfg.model.dropout

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.block = self._build_block()

    def _build_block(self):
        """Create the internal submodules for the encoder block.

        Returns:
          torch.nn.ModuleDict: Dictionary containing attention, normalization,
            and feed-forward layers.
        """
        return nn.ModuleDict(
            {
                "attn": nn.MultiheadAttention(
                    embed_dim=self.model_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                    device=self.device,
                    dtype=torch.float,
                ),
                "norm1": nn.LayerNorm(
                    self.model_dim,
                    device=self.device,
                    dtype=torch.float,
                ),
                "ffn": nn.Sequential(
                    nn.Linear(
                        self.model_dim,
                        4 * self.model_dim,
                        device=self.device,
                        dtype=torch.float,
                    ),
                    self._activation(self.activation_function),
                    nn.Linear(
                        4 * self.model_dim,
                        self.model_dim,
                        device=self.device,
                        dtype=torch.float,
                    ),
                ),
                "norm2": nn.LayerNorm(
                    self.model_dim,
                    device=self.device,
                    dtype=torch.float,
                ),
            }
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

    def forward(self, hidden, mask=None):
        """Apply the encoder block to a sequence of hidden states.

        Parameters:
          hidden: Tensor of shape (batch, seq_len, features).
          mask: Tensor of shape (batch, seq_len).

        Returns:
          torch.Tensor: Updated hidden states after self-attention and the
            feed-forward network.

        Raises:
          ValueError: If the input tensor is not three-dimensional.
        """
        if hidden.dim() != 3:
            raise ValueError(
                "Expected input of shape (batch, seq_len, features); "
                f"got {tuple(hidden.shape)}"
            )

        attn_out, _ = self.block["attn"](
            hidden, hidden, hidden, key_padding_mask=mask, need_weights=False
        )
        hidden = self.block["norm1"](hidden + attn_out)
        feed_forward = self.block["ffn"](hidden)
        hidden = self.block["norm2"](hidden + feed_forward)

        return hidden
