import torch
from torch.utils.data import Dataset

from utils.DatasetUtils import generate_info
from utils.SeedManager import SeedManager
from utils.StableSplineGenerator import (
    generate_stable_spline,
)


class KernelParametersPredictionInputGAlphaOutputC(Dataset):
    def __init__(self, cfg, batch_size):
        """Generates a dataset predicting the 'c' parameter given the impulse response 'g' and 'alpha' (or 'eta')."""
        self.batch_size = (
            batch_size  # TODO: Implement a more robust batch size handling mechanism
        )
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
        self.kernel = cfg.dataset.kernel

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __getitem__(self, idx):
        # Sample the hyperparameters governing the system physics
        alpha = (self.alpha_max - self.alpha_min) * SeedManager().dataset_rand(
            1
        ).item() + self.alpha_min
        c = (self.c_max - self.c_min) * SeedManager().dataset_rand(
            1
        ).item() + self.c_min
        eta = (self.eta_max - self.eta_min) * SeedManager().dataset_rand(
            1
        ).item() + self.eta_min

        # Generate the synthetic system identification problem
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
            kernel=self.kernel,
        )

        g = sample["g"]

        param1 = alpha if self.kernel == "stable_spline" else eta
        param1_tensor = torch.tensor([param1], dtype=torch.float)

        input = torch.cat([g, param1_tensor], dim=0).unsqueeze(-1)

        target = torch.tensor([c], dtype=torch.float)

        return input, target, generate_info(sample)
