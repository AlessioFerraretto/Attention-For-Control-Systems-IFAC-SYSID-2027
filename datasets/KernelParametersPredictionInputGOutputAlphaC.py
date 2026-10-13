import torch
from torch.utils.data import Dataset

from utils.DatasetUtils import generate_info
from utils.SeedManager import SeedManager
from utils.StableSplineGenerator import (
    generate_stable_spline,
)


class KernelParametersPredictionInputGOutputAlphaC(Dataset):
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

        g = sample["g"]

        input = g
        input = input.unsqueeze(-1)

        target = torch.tensor([alpha, c], dtype=torch.float)

        return input, target, generate_info(sample)
