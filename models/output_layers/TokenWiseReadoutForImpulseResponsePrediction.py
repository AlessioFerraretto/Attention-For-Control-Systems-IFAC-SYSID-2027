"""Token-wise classifier for impulse response prediction tasks.

This module applies a shared two-layer feedforward network with a GELU activation
independently to each temporal token of the input sequence.

This maintains temporal structure and drastically reduces parameter count compared
to a flattened readout approach.
"""

import torch
from torch import nn

from monitor.TrackableModule import TrackableModule


class TokenWiseReadoutForImpulseResponse(TrackableModule):
    """Build a token-wise feed-forward readout for impulse response prediction.

    The readout consumes a temporal window of features and produces a single
    scalar prediction for each token step independently.
    """

    def __init__(
        self,
        cfg,
        device: torch.device,
    ):
        """Initialize the token-wise readout.

        Parameters:
          cfg: Configuration object containing classifier and dataset settings.
          device: Torch device on which the module parameters should live.
        """
        super().__init__()

        # The input dimension matches the Transformer's model_dim
        self.input_dim = cfg.model.input_dim_classifier
        self.hidden_dim = cfg.model.hidden_dim_classifier
        self.output_dim = cfg.model.output_dim_classifier
        self.out_bias1 = cfg.model.out_bias1_classifier
        self.out_bias2 = cfg.model.out_bias2_classifier
        self.g_length = cfg.dataset.g_length

        # The final layer projects to 1 because each token predicts a single
        # point of the impulse response.
        self.classifier = nn.Sequential(
            nn.Linear(
                self.input_dim, self.hidden_dim, bias=self.out_bias1, device=device
            ),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim, self.output_dim, bias=self.out_bias2, device=device
            ),
        )

    def forward(self, u):
        """Produce predictions and logits processing each token independently.

        Parameters:
          u: Input tensor of shape (batch, seq_len, features).

        Returns:
          tuple[torch.Tensor, torch.Tensor]: A pair containing predictions of
            shape (batch, g_length, 1) and logits of shape
            (batch, g_length).
        """

        # 1. Isolate the final g_length tokens
        # Shape: (batch, g_length, input_dim)
        u_sliced = u[:, -self.g_length :, :]

        # 2. Apply the neural network. PyTorch maps Linear over the last dimension.
        # Each token is processed in parallel by the exact same weights.
        # Shape: (batch, g_length, 1)
        output = self.classifier(u_sliced)

        # 3. Remove the last dummy dimension to obtain scalar logits
        # Shape: (batch, g_length)
        predictions = output.squeeze(-1) if self.output_dim == 1 else output

        return predictions, predictions
