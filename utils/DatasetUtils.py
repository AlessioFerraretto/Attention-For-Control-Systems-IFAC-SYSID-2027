import torch

from utils.StableSplineGenerator import (
    KERNELS,
)


def generate_info(sample):
    u_past = sample["u_past"]
    y_past = sample["y_past"]
    u_future = sample["u_future"]
    y_future = sample["y_future"]
    g = sample["g"]
    snr = sample["snr"]
    c = sample["c"]
    alpha = sample["alpha"]
    eta = sample["eta"]
    g_length = sample["g_length"]
    kernel = sample["kernel"]
    noise_variance = sample["noise_variance"]

    u = torch.cat([u_past, u_future], dim=0)
    y = torch.cat([y_past, y_future], dim=0)

    K = KERNELS[kernel](g_length, alpha=alpha, eta=eta, c=c)

    info = {
        "u_past": u_past.float(),
        "u_future": u_future.float(),
        "u": u.float(),
        "y_past": y_past.float(),
        "y_future": y_future.float(),
        "y": y.float(),
        "g": g.float(),
        "snr": snr,
        "c": torch.tensor(c, dtype=torch.float),
        "alpha": torch.tensor(alpha, dtype=torch.float),
        "eta": torch.tensor(eta, dtype=torch.float),
        "noise_variance": noise_variance.float(),
        "kernel_matrix": K.float(),
        "g_length": g_length,
        "kernel": kernel,
    }

    return info
