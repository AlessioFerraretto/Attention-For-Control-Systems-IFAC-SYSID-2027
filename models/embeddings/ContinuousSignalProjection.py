import torch
from torch import nn

from monitor.TrackableModule import TrackableModule

# TODO: Check if this is correct


class ContinuousSignalProjection(TrackableModule):
    def __init__(self):
        super().__init__()
        self.myEmbedding = nn.Identity()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # TODO: Check if this is correct

        return self.myEmbedding(u)
