"""
Uses a fixed system (alpha, c) given.
Fixes (g) using the given system (alpha, c).
Dynamically generates u and y given the fixed (g)
The generator output is of the form [u_past, y_past, u_future, padding] -> [y_future].
"""

import torch
from torch.utils.data import Dataset

from utils.StableSplineGenerator import (
    sample_system_impulse_response,
    simulate_io_data,
)


class FixedFuturePredictionV1(Dataset):
    def __init__(self, cfg, batch_size, alpha, c):
        """Generates a static dataset of synthetic linear systems."""
        self.batch_size = batch_size  # TODO: Find a more elegant way
        self.batches_per_epoch = cfg.dataset.batches_per_epoch

        self.g_length = cfg.dataset.g_length
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
        self.low_pass_cutoff = cfg.dataset.low_pass_cutoff
        self.low_pass_cutoff_order = cfg.dataset.low_pass_cutoff_order

        # Fixes (g) using the given system (alpha, c)
        self.fixed_g, self.system_params = sample_system_impulse_response(
            self.g_length,
            self.c,
            self.alpha,
            self.jitter_regularization_parameter,
        )

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __getitem__(self, idx):
        n_total = self.n_past + self.n_future

        # Generate (u, y) using the fixed (g)
        u, y, snr = simulate_io_data(
            self.fixed_g,
            n_total,
            self.snr_min,
            self.snr_max,
            self.a_param,
        )

        # Splitting
        u_past = u[: self.n_past]
        y_past = y[: self.n_past]
        u_future = u[self.n_past :]
        y_future = y[self.n_past :]

        padding_tensor = torch.full(
            size=(self.n_future,),
            fill_value=self.padding,
            dtype=torch.float,
        )

        input = torch.cat(
            [
                u_past,
                y_past,
                u_future,
                padding_tensor,
            ]
        )
        input = input.unsqueeze(-1)

        target = y_future
        target = target.unsqueeze(-1)

        return input, target
