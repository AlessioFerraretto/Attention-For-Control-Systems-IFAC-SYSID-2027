from collections.abc import Callable

import torch


class InductionHead:
    def loss_fn(self) -> Callable:
        # return torch.nn.CrossEntropyLoss()
        return torch.nn.MSELoss()  # TODO: Check if this is correct

    def compute_metrics(self, predictions, targets):
        """
        TODO: Write description

        Parameters:
            predictions : torch.Tensor
                Predicted token indices with shape (batch, seq_len, ...).
            targets : torch.Tensor
                Ground-truth token indices with shape compatible with `predictions`.

            Returns:
            dict: A similarity-style score where smaller absolute error yields
            a larger value.
        """

        # TODO: Check if this is correct
        # Update: this is not correct, we may want to use R^2
        # or another statistical method

        distances = torch.abs(predictions - targets)

        low_errors = torch.all(distances < 0.5, dim=1)

        total_score = torch.sum(low_errors).item()

        return {"correct_sequences": total_score}
