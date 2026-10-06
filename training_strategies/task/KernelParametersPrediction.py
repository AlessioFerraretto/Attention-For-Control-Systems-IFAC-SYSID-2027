from collections.abc import Callable

import torch

from models.ScaleInvariantLoss import ScaleInvariantLoss
from utils.Estimators import compute_mmse
from utils.StableSplineGenerator import BATCHED_KERNELS


class KernelParametersPrediction:
    def __init__(self, cfg):
        self.requires_y_in_input = False
        self.kernel_type = cfg.dataset.kernel

    def loss_fn(self) -> Callable:
        return ScaleInvariantLoss()

    def compute_metrics(self, predictions, targets, info):
        """
        Computes the FIT metric by projecting the predicted kernel parameters
        into the impulse response space and comparing it with the true 'g'.
        """
        # Collapse temporal dimension if token-wise readout is used
        p = predictions.mean(dim=1) if predictions.dim() == 3 else predictions

        param1 = p[:, 0]
        c_param = p[:, 1]

        u_batch = info["u"].to(predictions.device)
        y_batch = info["y"].to(predictions.device)
        true_g = info["g"].to(predictions.device)
        noise_var_batch = info["noise_variance"].to(predictions.device)
        g_length = int(info["g_length"][0].item())

        # Generate K_hat using the batched methods
        K_hat = BATCHED_KERNELS[self.kernel_type](g_length, c_param, param1)

        # Compute estimated impulse response from predicted kernel parameters
        g_hat, fit_g_batch = compute_mmse(
            u_batch, y_batch, true_g, noise_var_batch, K_hat, g_length
        )

        mean_fit = fit_g_batch.mean()

        return {"fit": mean_fit.item()}

    def compute_y_hat(self, preds, u, y_with_grad, noise_var, g_length):
        """Computes the estimated output y_hat using the predicted kernel parameters and the input u."""
        B, n = y_with_grad.shape

        if preds.dim() == 3:
            preds = preds.mean(dim=1)

        param1 = preds[:, 0]
        c_param = preds[:, 1]

        K = BATCHED_KERNELS[self.kernel_type](g_length, c_param, param1)

        u_padded = torch.nn.functional.pad(u, (g_length - 1, 0))
        U = u_padded.unfold(dimension=1, size=g_length, step=1)
        U = torch.flip(U, dims=[2])

        UK = torch.bmm(U, K)
        UKUT = torch.bmm(UK, U.transpose(1, 2))

        I = torch.eye(n, device=y_with_grad.device).unsqueeze(0).expand(B, n, n)
        sigma2_I = noise_var.view(B, 1, 1) * I

        LHS = UKUT + sigma2_I
        x = torch.linalg.solve(LHS, y_with_grad.unsqueeze(-1))
        y_hat = torch.bmm(UKUT, x).squeeze(-1)

        return y_hat
