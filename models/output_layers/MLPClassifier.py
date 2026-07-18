"""
Multi-layer perceptron (MLP) classifier.

This module applies a two-layer feedforward network with a GELU activation
to map input features directly to output classes.

Unlike sequence-based classifiers, this module assumes the input is already
a fixed-size representation (no pooling is performed).
"""

import torch
from torch import nn

from monitor.TrackableModule import TrackableModule


class MLPClassifier(TrackableModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        out_bias1: bool,
        out_bias2: bool,
        device: torch.device,
    ):
        super().__init__()

        # Two-layer feedforward classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, out_bias1, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, out_bias2, device=device),
        )

    def forward(self, u):
        """
        Parameters
            u: Input features [B, D]
            embedding: Unused (kept for interface consistency)

        Returns:
            predictions: [B]
            logits: [B, output_dim]
        """

        logits = self.classifier(u)

        # Predicted class indices
        predictions = torch.argmax(logits, dim=-1)

        return predictions, logits
