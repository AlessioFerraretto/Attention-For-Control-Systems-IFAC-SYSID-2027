import torch
from torch.utils.data import Dataset

from utils.SeedManager import SeedManager
from utils.StableSplineGenerator import (
    KERNELS,
    generate_stable_spline,
)


class KernelParametersPredictionInputUYCOutputAlpha(Dataset):
    def __init__(self, cfg, batch_size):
        """Generates a dataset predicting 'alpha' (or 'eta') given 'u', 'y' and 'c'."""
        self.batch_size = batch_size
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
        self.kernel_type = cfg.dataset.kernel

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __getitem__(self, idx):
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
            kernel=self.kernel_type,
        )

        u = torch.cat([sample["u_past"], sample["u_future"]], dim=0)
        y = torch.cat([sample["y_past"], sample["y_future"]], dim=0)
        g = sample["g"]
        snr = sample["snr"]

        uy = torch.stack([u, y], dim=-1)

        # 1. Create the prompt token containing the known parameter c
        c_tensor = torch.tensor([[c, 0.0]], dtype=torch.float)

        # 2. Create the padding token
        padding_tensor = torch.full(
            size=(2, 2), fill_value=self.padding, dtype=torch.float
        )

        # 3. Concatenate the prompt, input data, and padding tokens
        input = torch.cat([uy, c_tensor, padding_tensor], dim=0)

        # 4. The target contains the unknown parameter alpha or eta
        param1 = alpha if self.kernel_type == "stable_spline" else eta
        target = torch.tensor([param1], dtype=torch.float)

        var_y = torch.var(y, unbiased=False)
        noise_variance = var_y / (snr + 1.0)
        K = KERNELS[self.kernel_type](self.g_length, alpha=alpha, eta=eta, c=c)

        info = {
            "u": u.float(),
            "y": y.float(),
            "g": g.float(),
            "c": torch.tensor(c, dtype=torch.float),
            "alpha": torch.tensor(alpha, dtype=torch.float),
            "eta": torch.tensor(eta, dtype=torch.float),
            "noise_variance": noise_variance.float(),
            "kernel_matrix": K.float(),
            "g_length": self.g_length,
            "kernel_type": self.kernel_type,
        }

        return input, target, info
