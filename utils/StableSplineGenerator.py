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


def sample_system_impulse_response(g_length, c, alpha, jitter_regularization_parameter):
    """Sample the stable spline hyperparameters and impulse response 'g'.

    Call this ONCE per system you want to identify (e.g., in Dataset __init__)
    to ensure the underlying physics of the system remain constant.
    """
    K_eta = generate_stable_spline_kernel(g_length, c, alpha)
    jitter = torch.eye(g_length) * jitter_regularization_parameter

    g = SeedManager().dataset_multivariate_normal(K_eta + jitter)

    return g, {"c": c, "alpha": alpha}


def simulate_io_data(g, n, snr_min, snr_max, a_param):
    """Simulate input 'u' and noisy output 'y' given a fixed impulse response 'g'.

    Call this multiple times (e.g., in Dataset __getitem__) to generate
    different dynamic responses for the same physical system.
    """
    g_length = len(g)

    # Generate input sequence u_t
    if a_param is not None:
        # Generate white noise with NumPy
        u_white = SeedManager().dataset_numpy_randn(n)

        # Define continuous-time transfer funcsimulate_io_datation H(s) = 1 / (s - a_param)
        num = [1.0]
        den = [1.0, -a_param]

        # Convert to discrete-time using the bilinear transform
        # dt is the sampling period.
        dt = 1.0
        b_discrete, a_discrete, _ = signal.cont2discrete(
            (num, den), dt, method="bilinear"
        )

        # Extract the discrete filter coefficients
        b = b_discrete[0]
        a = a_discrete[0]

        # Apply the filter
        # (Using filtfilt keeps it zero-phase like your original code.
        #  If you want a causal, real-time filter instead, use signal.lfilter)
        u_filtered = signal.filtfilt(b, a, u_white)

        # Convert back to a PyTorch tensor
        u = torch.tensor(u_filtered.copy(), dtype=torch.float)
    else:
        u = SeedManager().dataset_randn(n)

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
    snr_min,
    snr_max,
    jitter_regularization_parameter,
    a_param,
):
    """Generate a synthetic system identification problem.

    (Retained for backward compatibility. Wraps the modular functions.)
    """
    n = n_past + n_future

    # 1. Sample the system physics
    g, system_params = sample_system_impulse_response(
        g_length, c, alpha, jitter_regularization_parameter
    )

    # 2. Simulate a single trajectory of length n
    u, y, snr = simulate_io_data(g, n, snr_min, snr_max, a_param)

    # 3. Split the sequences into past and future
    u_past = u[:n_past]
    y_past = y[:n_past]
    u_future = u[n_past:]
    y_future = y[n_past:]

    return {
        "u_past": u_past,
        "y_past": y_past,
        "u_future": u_future,
        "y_future": y_future,
        "g": g,
        "eta": {
            "c": system_params["c"],
            "alpha": system_params["alpha"],
            "snr": snr,
        },
    }
