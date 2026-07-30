"""
Uses a fixed system (alpha, c) given.
Fixes (g) using the given system (alpha, c).
Dynamically generates u and y given the fixed (g)
The generator output is of the form [u, padding] -> [y].
"""

import torch
from torch.utils.data import Dataset

from utils.StableSplineGenerator import (
    sample_system_impulse_response,
    simulate_io_data,
)


class FixedFuturePredictionV2(Dataset):
    def __init__(self, cfg, batch_size, alpha, c):
        """Generates a static dataset of synthetic linear systems."""
        self.batch_size = batch_size  # TODO: Find a more elegant way
        self.batches_per_epoch = cfg.dataset.batches_per_epoch

        self.g_length = cfg.dataset.g_length
        self.n = cfg.dataset.n
        self.c = c
        self.alpha = alpha
        self.snr_min = cfg.dataset.snr_min
        self.snr_max = cfg.dataset.snr_max
        self.jitter_regularization_parameter = (
            cfg.dataset.jitter_regularization_parameter
        )
        self.padding = cfg.dataset.padding
        self.input_dim = cfg.model.input_dim

        # Fixes (g) using the given system (alpha, c)
        self.fixed_g, self.system_params = sample_system_impulse_response(
            g_length=self.g_length,
            c=self.c,
            alpha=self.alpha,
            jitter_regularization_parameter=self.jitter_regularization_parameter,
        )

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __getitem__(self, idx):
        # Dynamically generate u and y using the fixed system
        u, y, snr_db = simulate_io_data(
            g=self.fixed_g, n=self.n, snr_min=self.snr_min, snr_max=self.snr_max
        )

        padding_tensor = torch.full(
            size=(self.n,),
            fill_value=self.padding,
            dtype=torch.float,
        )

        input = torch.cat([u, padding_tensor])
        input = input.unsqueeze(-1)

        target = y
        target = target.unsqueeze(-1)

        return input, target
