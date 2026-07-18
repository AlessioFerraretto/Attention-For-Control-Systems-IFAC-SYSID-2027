"""Complete Neural Network composition module."""

from torch import nn

from models.ResidualBlock import ResidualBlock
from monitor.TrackableModule import TrackableModule


class NNComposition(TrackableModule):
    """Complete neural network model composition.

    Combines embedding, cascade of residual blocks, and output module
    into a unified model pipeline.
    input -> Embedding -> Cascade of Residual Blocks -> RMS normalization layer -> output linear layer to produce logits -> logits

    Attributes:
      embedding : nn.Embedding
        Embedding layer.
      layers : nn.ModuleList
        List of residual block layers.
      output : nn.Module
        Output module for final predictions.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        layers: list[ResidualBlock],
        output_layer: nn.Module,
    ):
        super().__init__()

        self.embedding = embedding
        self.layers = nn.ModuleList(layers)
        self.output = output_layer

    def forward(self, u):
        """Process input through embedding and residual layers.

        Applies embedding lookup, passes through cascade of residual blocks,
        then applies output layer for final predictions.

        Parameters:
          u : torch.Tensor
            Input token indices of shape (batch_size, sequence_length).

        Returns:
          Predictions and logits from the output layer.
        """
        u = self.embedding(u)

        for layer in self.layers:
            u = layer(u)

        # Pass the embedding layer instance, in case the output requires it
        return self.output(u)
