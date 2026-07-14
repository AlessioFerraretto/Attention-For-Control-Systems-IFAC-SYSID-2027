from collections.abc import Callable

import torch


class InductionHead:
    def loss_fn(self) -> Callable:
        return torch.nn.CrossEntropyLoss()

    def compute_metrics(self, predictions, labels):
        """

        Parameters:
            predictions : torch.Tensor
                Predicted token indices with shape (batch, seq_len, ...).
            labels : torch.Tensor
                Ground-truth token indices with shape compatible with `predictions`.
            batch_size : int
                Number of sequences in the batch used to normalize the result.

        Returns:
        """

        return {"metrics": "No metrics yet implemented for InductionHead dataset."}
