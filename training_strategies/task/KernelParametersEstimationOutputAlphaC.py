from collections.abc import Callable

import torch

from models.ScaleInvariantLoss import ScaleInvariantLoss
from utils.Estimators import build_matrix_U_batched, compute_mmse
from utils.StableSplineGenerator import BATCHED_KERNELS


class KernelParametersEstimationOutputAlphaC:
    def __init__(self, cfg):
        self.requires_y_in_input = False
        self.kernel_type = cfg.dataset.kernel

    def loss_fn(self) -> Callable:
        return ScaleInvariantLoss()

    def _prepare_kernel_and_data(self, predictions, info):
        if predictions.dim() == 3:
            # Extract only the last 2 tokens (corresponding to the padding_tensor)
            predictions = predictions[:, -2:, :].mean(dim=1)

        alpha = eta = predictions[:, 0]
        c = predictions[:, 1]

        u_batch = info["u"].to(predictions.device)
        y_batch = info["y"].to(predictions.device)
        true_g = info["g"].to(predictions.device)
        noise_var_batch = info["noise_variance"].to(predictions.device)
        g_length = int(info["g_length"][0].item())

        # Generate K_hat using the batched methods
        K_hat = BATCHED_KERNELS[self.kernel_type](g_length, alpha=alpha, eta=eta, c=c)

        return K_hat, u_batch, y_batch, true_g, noise_var_batch, g_length

    def compute_metrics(self, predictions, targets, info):
        """
        Computes the FIT metric by projecting the predicted kernel parameters
        into the impulse response space and comparing it with the true 'g'.
        """
        K_hat, u_batch, y_batch, true_g, noise_var_batch, g_length = (
            self._prepare_kernel_and_data(predictions, info)
        )

        # Compute estimated impulse response from predicted kernel parameters
        g_hat, fit_g_batch = compute_mmse(
            u_batch, y_batch, true_g, noise_var_batch, K_hat, g_length
        )

        mean_fit = fit_g_batch.mean()

        return {"fit": mean_fit.item()}

    def compute_y_hat(self, predictions, info, u, y_with_grad):
        """Computes the estimated output y_hat using the predicted kernel parameters and the input u."""
        K, u_batch, y_batch, true_g, noise_var_batch, g_length = (
            self._prepare_kernel_and_data(predictions, info)
        )

        B, n = y_with_grad.shape

        U = build_matrix_U_batched(u, g_length)

        UK = torch.bmm(U, K)
        UKUT = torch.bmm(UK, U.transpose(1, 2))

        I = torch.eye(n, device=y_with_grad.device).unsqueeze(0).expand(B, n, n)
        sigma2_I = noise_var_batch.view(B, 1, 1) * I

        LHS = UKUT + sigma2_I
        x = torch.linalg.solve(LHS, y_with_grad.unsqueeze(-1))
        y_hat = torch.bmm(UKUT, x).squeeze(-1)

        return y_hat
