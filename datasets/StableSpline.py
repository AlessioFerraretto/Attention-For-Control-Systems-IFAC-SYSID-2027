import torch
from torch.utils.data import Dataset

from utils.StableSplineGenerator import generate_stable_spline


class StableSpline(Dataset):
    def __init__(self, cfg, batch_size):
        """Generates a static dataset of synthetic linear systems."""
        self.batch_size = batch_size  # TODO: Find a more elegant way
        self.batches_per_epoch = cfg.dataset.batches_per_epoch

        self.g_length = cfg.dataset.g_length
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

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __getitem__(self, idx):
        sample = generate_stable_spline(
            g_length=self.g_length,
            n_past=self.n_past,
            n_future=self.n_future,
            c_min=self.c_min,
            c_max=self.c_max,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            snr_min=self.snr_min,
            snr_max=self.snr_max,
            jitter_regularization_parameter=self.jitter_regularization_parameter,
        )
        u = torch.cat([sample["u_past"], sample["u_future"]])
        y = torch.cat([sample["y_past"], sample["y_future"]])
        g = sample["g"]

        # Stack u and y as features for the sequence: shape (seq_len, 2)
        input = torch.stack([u, y], dim=1).float()
        target = g.float()

        return input, target
