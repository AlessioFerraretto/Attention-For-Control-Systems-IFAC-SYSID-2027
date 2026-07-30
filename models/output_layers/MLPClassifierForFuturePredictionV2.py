"""MLP classifier for future prediction tasks.

This module applies a two-layer feedforward network with a GELU activation to
map input features directly to output classes.

Unlike sequence-based classifiers, this module assumes the input is already a
fixed-size representation and does not perform pooling.
"""

import torch
from torch import nn

from monitor.TrackableModule import TrackableModule


class MLPClassifierForFuturePredictionV2(TrackableModule):
    """Build a feed-forward classifier for future prediction outputs.

    The classifier consumes a flattened window of past and future features and
    produces logits for the requested output dimension.
    """

    def __init__(
        self,
        cfg,
        device: torch.device,
    ):
        """Initialize the classifier.

        Parameters:
          cfg: Configuration object containing classifier and dataset settings.
          device: Torch device on which the module parameters should live.
        """
        super().__init__()

        self.input_dim = cfg.model.input_dim_classifier
        self.hidden_dim = cfg.model.hidden_dim_classifier
        self.output_dim = cfg.model.output_dim_classifier
        self.out_bias1 = cfg.model.out_bias1_classifier
        self.out_bias2 = cfg.model.out_bias2_classifier
        self.n_past = cfg.dataset.n_past
        self.n_future = cfg.dataset.n_future

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, self.out_bias1, device=device),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim, self.out_bias2, device=device),
        )

    def forward(self, u):
        """Produce predictions and logits from a sequence window.

        Parameters:
          u: Input tensor of shape (batch, seq_len, features).

        Returns:
          tuple[torch.Tensor, torch.Tensor]: A pair containing predictions of
            shape (batch, n_future, 1) and logits of shape
            (batch, n_future).
        """
        # 1. Slice along the sequence length dimension L (dim=1)
        u_sliced = u[:, self.n_past : self.n_past * 2 + self.n_future, :]

        u_flattened = torch.flatten(u_sliced, start_dim=1)

        # 2. Get the continuous output values (shape: [batch, n_future])
        logits = self.classifier(u_flattened)

        # 3. Create the predictions
        # Since this is a regression task, the raw outputs are the predictions.
        # Add the last dimension to match the required shape [batch, n_future, 1]
        predictions = logits.unsqueeze(-1)

        return predictions, logits
