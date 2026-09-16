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


class KernelParametersPredictionV4(Dataset):
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

        sample = generate_stable_spline(
            self.g_length,
            self.n_past,
            self.n_future,
            c,
            alpha,
            self.snr_min,
            self.snr_max,
            self.jitter_regularization_parameter,
            self.a_param,
        )

        g = sample["g"]

        padding_tensor = torch.full(
            size=(2,),
            fill_value=self.padding,
            dtype=torch.float,
        )

        alpha_tensor = torch.tensor([alpha], dtype=torch.float)

        input = torch.cat([g, alpha_tensor, padding_tensor])
        input = input.unsqueeze(-1)

        target = torch.tensor([c], dtype=torch.float)

        return input, target
