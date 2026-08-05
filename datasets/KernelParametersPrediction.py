"""
Uses a fixed system (alpha, c) given.
Fixes (g) using the given system (alpha, c).
Dynamically generates u and y given the fixed (g)
The generator output is of the form [[u],[y]] -> [alpha,c].
"""

import torch
from torch.utils.data import Dataset

from utils.StableSplineGenerator import (
    generate_stable_spline,
)


class KernelParametersPrediction(Dataset):
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
        self.snr_min = cfg.dataset.snr_min
        self.snr_max = cfg.dataset.snr_max
        self.jitter_regularization_parameter = (
            cfg.dataset.jitter_regularization_parameter
        )
        self.padding = cfg.dataset.padding
        self.input_dim = cfg.model.input_dim
        self.low_pass_cutoff = cfg.dataset.low_pass_cutoff
        self.low_pass_cutoff_order = cfg.dataset.low_pass_cutoff_order

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __getitem__(self, idx):
        # Generate (alpha, c) with which to generate (g) and the resulting (u, y) using the fixed system (c, alpha)
        alpha = (self.alpha_max - self.alpha_min) * torch.rand(
            1
        ).item() + self.alpha_min
        c = (self.c_max - self.c_min) * torch.rand(1).item() + self.c_min

        sample = generate_stable_spline(
            self.g_length,
            self.n_past,
            self.n_future,
            c,
            alpha,
            self.snr_min,
            self.snr_max,
            self.jitter_regularization_parameter,
            self.low_pass_cutoff,
            self.low_pass_cutoff_order,
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

        target = torch.tensor((alpha, c)).unsqueeze(-1)

        return input, target
