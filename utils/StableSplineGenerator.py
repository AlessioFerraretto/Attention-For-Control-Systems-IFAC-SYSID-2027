"""Stable spline data generation utilities.

This module provides helper functions for generating stable spline covariance
kernels and synthetic input-output datasets using a first-order stable spline
(TC) prior.
"""

import scipy.signal as signal
import torch
import torch.nn.functional as F

from utils.SeedManager import SeedManager


def generate_stable_spline_kernel(impulse_length, c, alpha):
    """Generate a first-order stable spline covariance matrix using PyTorch."""
    idx = torch.arange(1, impulse_length + 1)
    max_indices = torch.maximum(idx[:, None], idx)
    return c * (alpha**max_indices)


def sample_stable_spline_impulse_response(
    g_length, c, alpha, eta, jitter_regularization_parameter
):
    """Sample the stable spline hyperparameters and impulse response 'g'.

    Call this ONCE per system you want to identify (e.g., in Dataset __init__)
    to ensure the underlying physics of the system remain constant.
    """
    K_eta = generate_stable_spline_kernel(g_length, c, alpha)
    jitter = torch.eye(g_length) * jitter_regularization_parameter

    g = SeedManager().dataset_multivariate_normal(K_eta + jitter)

    return g, {"c": c, "alpha": alpha}


def generate_gaussian_kernel(impulse_length, c, eta):
    """Generate a Gaussian (RBF) covariance matrix using PyTorch.
    Formula: K_{i,j} = c * exp(-eta * (i-j)^2)
    """
    idx = torch.arange(1, impulse_length + 1, dtype=torch.float)
    dist_sq = (idx[:, None] - idx[None, :]) ** 2

    return c * torch.exp(-eta * dist_sq)


def sample_gaussian_impulse_response(
    g_length, c, alpha, eta, jitter_regularization_parameter
):
    """Sample the Gaussian kernel hyperparameters and impulse response 'g'."""
    K_eta = generate_gaussian_kernel(g_length, c, eta)
    jitter = torch.eye(g_length) * jitter_regularization_parameter

    g = SeedManager().dataset_multivariate_normal(K_eta + jitter)

    return g, {"c": c, "eta": eta}


def generate_stable_spline_kernel_batched(
    g_length,
    alpha_batch,
    c_batch,
):
    """Batched generation of first-order stable spline covariance matrix."""
    device = c_batch.device
    B = c_batch.shape[0]
    k_idx = torch.arange(1, g_length + 1, device=device)
    max_indices = torch.maximum(k_idx[:, None], k_idx).float()

    return c_batch.view(B, 1, 1) * (
        alpha_batch.view(B, 1, 1) ** max_indices.unsqueeze(0)
    )


def generate_gaussian_kernel_batched(g_length, eta_batch, c_batch):
    """Batched generation of Gaussian (RBF) covariance matrix."""
    device = c_batch.device
    B = c_batch.shape[0]
    idx = torch.arange(1, g_length + 1, dtype=torch.float, device=device)
    dist_sq = (idx[:, None] - idx[None, :]) ** 2

    return c_batch.view(B, 1, 1) * torch.exp(
        -eta_batch.view(B, 1, 1) * dist_sq.unsqueeze(0)
    )


BATCHED_KERNELS = {
    "stable_spline": generate_stable_spline_kernel_batched,
    "gaussian": generate_gaussian_kernel_batched,
}

KERNELS = {
    "stable_spline": sample_stable_spline_impulse_response,
    "gaussian": sample_gaussian_impulse_response,
}


def simulate_io_data(g, n, snr_min, snr_max, filter_type, a_param):
    """Simulate input 'u' and noisy output 'y' given a fixed impulse response 'g'.

    Parameters:
        g: torch.Tensor, the true impulse response.
        n: int, sequence length.
        snr_min: float, minimum SNR for noise generation.
        snr_max: float, maximum SNR for noise generation.
        a_param: float, pole location for the low-pass filter (e.g., 0.9).
        filter_type: str, type of input generation. Options:
            - "white": standard white noise.
            - "ar1": unnormalized AR(1) low-pass filter H(z) = 1 / (z - a).
            - "ar1_normalized": unit static gain AR(1) filter H(z) = (1 - a) / (z - a).
    """
    g_length = len(g)

    # 1. Generate input sequence
    match filter_type:
        case "white":
            u = SeedManager().dataset_randn(n)

        case "type_1":
            if a_param is None:
                raise ValueError("a_param must be provided for the 'type_1' filter.")

            u_white = SeedManager().dataset_numpy_randn(n)
            b_coeff = [0.0, 1.0]
            a_coeff = [1.0, -a_param]

            u_filtered = signal.lfilter(b_coeff, a_coeff, u_white)
            u = torch.tensor(u_filtered.copy(), dtype=torch.float)

        case "type_2":
            if a_param is None:
                raise ValueError("a_param must be provided for the 'type_2' filter.")

            u_white = SeedManager().dataset_numpy_randn(n)
            b_coeff = [0.0, 1.0 - a_param]
            a_coeff = [1.0, -a_param]

            u_filtered = signal.lfilter(b_coeff, a_coeff, u_white)
            u = torch.tensor(u_filtered.copy(), dtype=torch.float)

        case _:
            raise ValueError(
                f"Unknown filter_type: '{filter_type}'. "
                "Choose 'white', 'type_1', or 'type_2'."
            )

    # Compute noise-free output y^0 using convolution
    u_reshaped = u.view(1, 1, n)
    g_flipped = torch.flip(g, dims=[0]).view(1, 1, g_length)

    y0_full = F.conv1d(u_reshaped, g_flipped, padding=g_length - 1)
    y0 = y0_full[0, 0, :n]

    # Add measurement noise e_t
    # SNR is defined as the variance ratio: Var(y0) / Var(noise)
    snr = (snr_max - snr_min) * SeedManager().dataset_rand(1).item() + snr_min

    noise_variance = torch.var(y0, unbiased=False) / snr
    e = SeedManager().dataset_randn(n) * torch.sqrt(noise_variance)

    y = y0 + e

    return u, y, snr


def generate_stable_spline(
    g_length,
    n_past,
    n_future,
    c,
    alpha,
    eta,
    snr_min,
    snr_max,
    jitter_regularization_parameter,
    filter_type,
    a_param,
    kernel="stable_spline",
):
    """Generate a synthetic system identification problem.

    (Retained for backward compatibility. Wraps the modular functions.)
    """
    n = n_past + n_future

    # 1. Sample the system physics
    g, system_params = KERNELS.get(kernel)(
        g_length,
        c,
        alpha=alpha,
        eta=eta,
        jitter_regularization_parameter=jitter_regularization_parameter,
    )

    # 2. Simulate a single trajectory of length n
    u, y, snr = simulate_io_data(g, n, snr_min, snr_max, filter_type, a_param)

    # 3. Split the sequences into past and future
    u_past = u[:n_past]
    y_past = y[:n_past]
    u_future = u[n_past:]
    y_future = y[n_past:]

    eta_dict = {**system_params, "snr": snr}

    return {
        "u_past": u_past,
        "y_past": y_past,
        "u_future": u_future,
        "y_future": y_future,
        "g": g,
        "eta": eta_dict,
    }
