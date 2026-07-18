"""
Sequence classifier using mean pooling and linear projection.

This module:
1. Normalizes given inputs using RMSNorm
2. Aggregates sequence information via mean pooling
3. Applies a linear classification head

This is a simple and effective baseline for sequence-level classification tasks.
"""

import torch
from torch import nn

from models.RMSNorm import RMSNorm
from monitor.TrackableModule import TrackableModule


class MeanPooledClassifier(TrackableModule):
    def __init__(
        self,
        output_dim: int,
        d_model: int,
        head_bias: bool,
        eps: float,
        device: torch.device,
    ):
        super().__init__()

        # Normalization layer
        self.rmsnorm = RMSNorm(d_model, device=device, eps=eps)
        # Linear classification head
        self.head = nn.Linear(d_model, output_dim, head_bias, device=device)

    def forward(self, u):
        """
        Parameters
            u: Input [B, L, D]
            embedding: Unused (kept for interface consistency)

        Returns:
            predictions: [B]
            logits: [B, output_dim]
        """
        # Normalize
        u = self.rmsnorm(u)

        # Mean pooling over sequence dimension
        u = torch.mean(u, 1)

        # Regression logits
        logits = self.head(u)

        # Predicted values
        predictions = logits  # TODO: Check if this is correct

        return predictions, logits
