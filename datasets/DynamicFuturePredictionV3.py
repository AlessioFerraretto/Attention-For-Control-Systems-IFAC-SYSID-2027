"""
Uses a fixed system (alpha, c) given.
Dynamically generates g using the given system (alpha, c).
The generator output is of the form [ [u_past, y_past], [u_future, padding] ] -> [y_future].
"""

import torch
from torch.utils.data import Dataset

from utils.StableSplineGenerator import (
    generate_stable_spline,
)


class DynamicFuturePredictionV3(Dataset):
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
        )

        # Splitting
        u_past = sample["u_past"]
        y_past = sample["y_past"]
        u_future = sample["u_future"]
        y_future = sample["y_future"]

        padding_tensor = torch.full(
            size=(self.n_future,),
            fill_value=self.padding,
            dtype=torch.float,
        )

        # Stack into 2D features: [u, y] at each timestep
        past_input = torch.stack([u_past, y_past], dim=-1)

        # Shape becomes (n_future, 2)
        future_input = torch.stack([u_future, padding_tensor], dim=-1)

        # Concatenate along the time dimension
        input = torch.cat([past_input, future_input], dim=0)

        # Target is the ground truth y values
        target = y_future

        # Add the feature dimension to target
        target = target.unsqueeze(-1)

        return input, target
