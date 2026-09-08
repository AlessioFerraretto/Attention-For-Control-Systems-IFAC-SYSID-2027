from collections.abc import Callable

import torch


class FuturePrediction:
    def loss_fn(self) -> Callable:
        # return torch.nn.CrossEntropyLoss()
        return torch.nn.MSELoss()  # TODO: Check if this is correct

    def compute_metrics(self, predictions, targets):
        """
        Computes the Mean Squared Error (MSE) between predictions and targets.

        Parameters:
            predictions : torch.Tensor
                Predicted values with shape (batch, seq_len, ...).
            targets : torch.Tensor
                Ground-truth values with shape compatible with `predictions`.

        Returns:
            dict: A dictionary containing the MSE score. Smaller values indicate
            better performance.
        """
        # Cast to float to ensure accurate division and prevent issues
        # if the inputs are integer types (like token indices)
        predictions = predictions.float()
        targets = targets.float()

        # Calculate Mean Squared Error
        mse = torch.mean((predictions - targets) ** 2)

        relative_error = torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)
        absolute_error = torch.abs(predictions - targets)
        mae = torch.mean(absolute_error)
        rmse = torch.sqrt(mse)
        percentage_error = torch.mean(relative_error) * 100

        return {
            "mse": mse.item(),
            "mae": mae.item(),
            "rmse": rmse.item(),
            "mape": percentage_error.item(),
        }
