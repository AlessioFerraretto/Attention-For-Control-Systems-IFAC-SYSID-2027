"""
Uses a fixed system (alpha, c) given.
Fixes (g) using the given system (alpha, c).
Dynamically generates u and y given the fixed (g)
The generator output is of the form [[u],[y]] -> [alpha,c].
"""

import torch
from torch.utils.data import Dataset

from utils.SeedManager import SeedManager
from utils.StableSplineGenerator import (
    generate_stable_spline,
)


class KernelParametersPredictionV3(Dataset):
    def __init__(self, cfg, batch_size):
        """Generates a static dataset of synthetic linear systems."""
        self.batch_size = batch_size  # TODO: Find a more elegant way
        self.batches_per_epoch = cfg.dataset.batches_per_epoch

        self.g_length = cfg.dataset.g_length
        self.n = cfg.dataset.n
        self.n_past = cfg.dataset.n_past
        self.n_future = cfg.dataset.n_future
        self.c_min = cfg.dataset.c_min
        self.c_max = cfg.dataset.c_max
        self.alpha_min = cfg.dataset.alpha_min
        self.alpha_max = cfg.dataset.alpha_max
        self.eta_min = cfg.dataset.eta_min
        self.eta_max = cfg.dataset.eta_max
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
        # Generate (alpha, c) with which to generate (g) using the fixed system (c, alpha)
        alpha = (self.alpha_max - self.alpha_min) * SeedManager().dataset_rand(
            1
        ).item() + self.alpha_min
        c = (self.c_max - self.c_min) * SeedManager().dataset_rand(
            1
        ).item() + self.c_min
        eta = (self.eta_max - self.eta_min) * SeedManager().dataset_rand(
            1
        ).item() + self.eta_min

        sample = generate_stable_spline(
            self.g_length,
            self.n_past,
            self.n_future,
            c,
            alpha,
            eta,
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
            size=(2,),
            fill_value=self.padding,
            dtype=torch.float,
        )

        input = torch.cat([g, padding_tensor])
        input = input.unsqueeze(-1)

        target = torch.tensor([alpha, c], dtype=torch.float)

        # Build the info for the evaluation of the estimators

        # 1. Exact reconstruction of the noise variance.
        # Given that y = y_0 + e, and SNR = Var(y_0) / Var(e),
        # we can derive: Var(e) = Var(y) / (SNR + 1.0)
        snr = sample["eta"]["snr"]
        var_y = torch.var(y, unbiased=False)
        noise_variance = var_y / (snr + 1.0)

        # 2. Reconstruction of the TC Kernel Matrix
        # K(k,j) = c * alpha^max(k,j)
        k_idx = torch.arange(1, self.g_length + 1)
        max_indices = torch.maximum(k_idx[:, None], k_idx)
        kernel_matrix = c * (alpha**max_indices)

        info = {
            "u": u,
            "y": y,
            "g": g,
            "noise_variance": noise_variance,
            "kernel_matrix": kernel_matrix,
            "g_length": self.g_length,
        }

        return input, target, info
