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
        cfg,
        device: torch.device,
    ):
        super().__init__()

        # Normalization layer
        self.rmsnorm = RMSNorm(cfg.model.input_dim_classifier, device=device)
        # Linear classification head
        self.head = nn.Sequential(
            nn.Linear(
                cfg.model.input_dim_classifier,
                cfg.model.hidden_dim_classifier,
                cfg.model.head_bias,
                device=device,
            ),
            nn.ReLU(),
            nn.Linear(
                cfg.model.hidden_dim_classifier,
                cfg.model.output_dim_classifier,
                bias=cfg.model.head_bias,
                device=device,
            ),
        )

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
        # [B, L, D]
        u = self.rmsnorm(u)

        # Mean pooling over sequence dimension L
        # [B, L, D] -> [B, D]
        u = torch.mean(u, dim=1)

        # Regression logits through the 2-layer head
        # [B, D] -> [B, D/2] -> [B, output_dim]
        logits = self.head(u)

        # Predicted values for alpha and c
        predictions = logits

        return predictions, logits
