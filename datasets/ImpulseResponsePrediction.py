"""
Uses a fixed system (alpha, c) given.
Fixes (g) using the given system (alpha, c).
Dynamically generates u and y given the fixed (g)
The generator output is of the form [[u],[y]] -> [g].
"""

import torch
from torch.utils.data import Dataset

from utils.StableSplineGenerator import generate_stable_spline


class ImpulseResponsePrediction(Dataset):
    def __init__(self, cfg, batch_size, alpha, c):
        """Generates a static dataset of synthetic linear systems."""
        self.batch_size = batch_size  # TODO: Find a more elegant way
        self.batches_per_epoch = cfg.dataset.batches_per_epoch

        self.g_length = cfg.dataset.g_length
        self.n = cfg.dataset.n
        self.n_past = cfg.dataset.n_past
        self.n_future = cfg.dataset.n_future
        self.c = c
        self.alpha = alpha
        self.snr_min = cfg.dataset.snr_min
        self.snr_max = cfg.dataset.snr_max
        self.jitter_regularization_parameter = (
            cfg.dataset.jitter_regularization_parameter
        )
        self.padding = cfg.dataset.padding
        self.input_dim = cfg.model.input_dim
        self.filter_type = cfg.dataset.filter_type
        self.a_param = cfg.dataset.a_param

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __getitem__(self, idx):
        # Generate (g) and the resulting (u, y) using the fixed system (c, alpha)
        sample = generate_stable_spline(
            self.g_length,
            self.n_past,
            self.n_future,
            self.c,
            self.alpha,
            self.snr_min,
            self.snr_max,
            self.jitter_regularization_parameter,
            self.filter_type,
            self.a_param,
        )

        # Splitting
        u_past = sample["u_past"]
        y_past = sample["y_past"]
        u_future = sample["u_future"]
        y_future = sample["y_future"]
        g = sample["g"]

        u = torch.cat([u_past, u_future], dim=0)
        y = torch.cat([y_past, y_future], dim=0)

        padding_tensor = torch.full(
            size=(self.g_length, 2),
            fill_value=self.padding,
            dtype=torch.float,
        )

        uy = torch.stack([u, y], dim=-1)
        input = torch.cat([uy, padding_tensor], dim=0)

        target = g

        # Build the mask

        mask = torch.cat(
            [
                torch.zeros(u.shape[0], dtype=torch.bool),
                torch.ones(self.g_length, dtype=torch.bool),
            ],
            dim=0,
        )

        # Build the info for the evaluation of the estimators

        # 1. Exact reconstruction of the noise variance.
        # Given that y = y_0 + e, and SNR = Var(y_0) / Var(e),
        # we can derive: Var(e) = Var(y) / (SNR + 1.0)
        snr = sample["snr"]
        var_y = torch.var(y, unbiased=False)
        noise_variance = var_y / (snr + 1.0)

        # 2. Reconstruction of the TC Kernel Matrix
        # K(k,j) = c * alpha^max(k,j)
        k_idx = torch.arange(1, self.g_length + 1)
        max_indices = torch.maximum(k_idx[:, None], k_idx)
        kernel_matrix = self.c * (self.alpha**max_indices)

        info = {
            "u": u,
            "y": y,
            "noise_variance": noise_variance,
            "kernel_matrix": kernel_matrix,
            "g_length": self.g_length,
        }

        return input, target, info, mask
