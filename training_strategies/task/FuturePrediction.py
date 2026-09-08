from collections.abc import Callable

import torch


class FuturePrediction:
    def loss_fn(self) -> Callable:
        # return torch.nn.CrossEntropyLoss()
        return torch.nn.MSELoss()  # TODO: Check if this is correct

    def compute_metrics(self, predictions, targets):
        """
        Computes the FIT metric between predicted and true impulse responses.

        Parameters:
            predictions : torch.Tensor
                Predicted impulse responses with shape (batch_size, g_length).
            targets : torch.Tensor
                Ground-truth impulse responses with shape (batch_size, g_length).

        Returns:
            dict: A dictionary containing 'fit', the closer to 100 the better.
        """
        # Cast to float to ensure accurate computation
        predictions = predictions.float()
        targets = targets.float()

        # Calculate the FIT metric (batch-wise, then averaged)
        # Compute the L2 (Euclidean) norm along the sequence dimension (dim=-1)
        norm_diff = torch.linalg.vector_norm(targets - predictions, dim=-1)
        norm_true = torch.linalg.vector_norm(targets, dim=-1)

        # Add an epsilon (1e-8) to the denominator to avoid division by zero
        fit_per_sample = 100.0 * (1.0 - (norm_diff / (norm_true + 1e-8)))
        
        # Compute the mean FIT across the batch
        mean_fit = torch.mean(fit_per_sample)

        return {
            "fit": mean_fit.item()
        }
