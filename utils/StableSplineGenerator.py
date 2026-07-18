"""Stable spline data generation utilities.

This module provides helper functions for generating stable spline covariance
kernels and synthetic input-output datasets using a first-order stable spline
(TC) prior.

"""

import numpy as np
import scipy.signal as signal
import torch
import torch.nn.functional as F


def generate_stable_spline_kernel(impulse_length, c, alpha):
    """Generate a first-order stable spline covariance matrix using PyTorch."""
    # Create 1-based indices: [1, 2, ..., impulse_length] directly on the GPU/CPU
    idx = torch.arange(1, impulse_length + 1)

    # Broadcasting:
    # idx[:, None] is an (N, 1) column vector
    # idx is an (N,) row vector
    # torch.maximum broadcasts them into an (N, N) matrix of max(i, j)
    max_indices = torch.maximum(idx[:, None], idx)

    # Compute the entire kernel
    return c * (alpha**max_indices)


def generate_stable_spline_kernel_old(impulse_length, c, alpha):
    """Generate a first-order stable spline covariance matrix.

    The covariance matrix follows the TC kernel formulation with entries
    proportional to c * alpha^max(i, j), using 1-based indexing semantics.

    Parameters:
      impulse_length : int
        Length of the impulse response.
      c : float
        Scaling factor for the kernel variance.
      alpha : float
        Decay rate, where 0 < alpha < 1.

    Returns:
      numpy.ndarray: The generated covariance matrix of shape
        (impulse_length, impulse_length).
    """
    K = np.zeros((impulse_length, impulse_length))
    for i in range(impulse_length):
        for j in range(impulse_length):
            # TC Kernel formulation: c * alpha^max(i, j)
            # Note: 1-based indexing equivalent adapted for 0-based Python
            K[i, j] = c * (alpha ** max(i + 1, j + 1))
    return K


def generate_stable_spline_old(
    g_length,
    n_past,
    n_future,
    c_min,
    c_max,
    alpha_min,
    alpha_max,
    snr_min,
    snr_max,
    jitter_regularization_parameter,
):
    """Generate a synthetic system identification problem.

    This function samples a stable spline impulse response and simulates a
    noisy input-output sequence. The result is split into past (context) and
    future (prediction) segments.

    Parameters:
      g_length : int
        Length of the impulse response.
      n_past : int
        Length of the past input-output sequence (context horizon).
      n_future : int
        Length of the future input-output sequence (prediction horizon).
      c_min : float
        Minimum scaling factor for the stable spline kernel.
      c_max : float
        Maximum scaling factor for the stable spline kernel.
      alpha_min : float
        Minimum decay rate for the stable spline kernel.
      alpha_max : float
        Maximum decay rate for the stable spline kernel.
      snr_min : float
        Minimum target signal-to-noise ratio in decibels.
      snr_max : float
        Maximum target signal-to-noise ratio in decibels.
    jitter_regularization_parameter: float
        Small positive value added to the diagonal of the covariance matrix
        for numerical stability during Cholesky decomposition.

    Returns:
      dict: A dictionary containing the generated dataset and parameters.
        "u_past" (numpy.ndarray): Past input sequence.
        "y_past" (numpy.ndarray): Past output sequence.
        "u_future" (numpy.ndarray): Future input sequence.
        "y_future" (numpy.ndarray): Future output sequence.
        "g" (numpy.ndarray): Sampled impulse response.
        "eta" (dict): Sampled hyperparameters.
    """

    # Total length of the sequence to simulate
    n = n_past + n_future

    # 1. Sample hyperparameters eta = [c, alpha]
    c = np.random.uniform(
        c_min,
        c_max,
    )
    alpha = np.random.uniform(alpha_min, alpha_max)  # Higher alpha = slower decay

    # Generate the Kernel K_eta
    K_eta = generate_stable_spline_kernel_old(g_length, c, alpha)

    # 2. Sample an impulse response g ~ N(0, K_eta)
    # Add a tiny jitter to the diagonal for numerical stability (Cholesky decomposition)
    jitter = np.eye(g_length) * jitter_regularization_parameter
    g = np.random.multivariate_normal(np.zeros(g_length), K_eta + jitter)

    # 3. Generate input sequence u_t (e.g., white Gaussian noise)
    u = np.random.normal(0, 1, n)

    # 4. Compute noise-free output y^0 using convolution
    # signal.convolve computes the sum_{k=1}^{m} g_k u_{t-k}
    y0 = signal.convolve(u, g, mode="full")[:n]

    # 5. Add measurement noise e_t
    # Let's target a specific Signal-to-Noise Ratio (SNR)
    snr_db = np.random.uniform(
        snr_min, snr_max
    )  # Varies between snr_min dB (noisy) and snr_max dB (clean)
    snr_linear = 10 ** (snr_db / 10)
    noise_variance = np.var(y0) / snr_linear
    e = np.random.normal(0, np.sqrt(noise_variance), n)

    y = y0 + e

    # 6. Split the sequences into past and future ---
    u_past = u[:n_past]
    y_past = y[:n_past]

    u_future = u[n_past:]
    y_future = y[n_past:]

    # 7. Store and return the dictionary matching your complex format
    return {
        "u_past": u_past,
        "y_past": y_past,
        "u_future": u_future,
        "y_future": y_future,
        "g": g,
        "eta": {"c": c, "alpha": alpha, "snr_db": snr_db},
    }


def generate_stable_spline(
    g_length,
    n_past,
    n_future,
    c_min,
    c_max,
    alpha_min,
    alpha_max,
    snr_min,
    snr_max,
    jitter_regularization_parameter,
):
    """Generate a synthetic system identification problem.

    This function samples a stable spline impulse response and simulates a
    noisy input-output sequence. The result is split into past (context) and
    future (prediction) segments.

    Parameters:
      g_length : int
        Length of the impulse response.
      n_past : int
        Length of the past input-output sequence (context horizon).
      n_future : int
        Length of the future input-output sequence (prediction horizon).
      c_min : float
        Minimum scaling factor for the stable spline kernel.
      c_max : float
        Maximum scaling factor for the stable spline kernel.
      alpha_min : float
        Minimum decay rate for the stable spline kernel.
      alpha_max : float
        Maximum decay rate for the stable spline kernel.
      snr_min : float
        Minimum target signal-to-noise ratio in decibels.
      snr_max : float
        Maximum target signal-to-noise ratio in decibels.
    jitter_regularization_parameter: float
        Small positive value added to the diagonal of the covariance matrix
        for numerical stability during Cholesky decomposition.

    Returns:
      dict: A dictionary containing the generated dataset and parameters.
        "u_past" (numpy.ndarray): Past input sequence.
        "y_past" (numpy.ndarray): Past output sequence.
        "u_future" (numpy.ndarray): Future input sequence.
        "y_future" (numpy.ndarray): Future output sequence.
        "g" (numpy.ndarray): Sampled impulse response.
        "eta" (dict): Sampled hyperparameters.
    """

    # Total length of the sequence to simulate
    n = n_past + n_future

    # 1. Sample hyperparameters eta = [c, alpha]
    # We use torch.rand(1).item() to generate python floats so they play nicely with your dictionary later
    c = (c_max - c_min) * torch.rand(1).item() + c_min
    alpha = (alpha_max - alpha_min) * torch.rand(
        1
    ).item() + alpha_min  # Higher alpha = slower decay

    # Generate the Kernel K_eta.
    K_eta = generate_stable_spline_kernel(g_length, c, alpha)

    # 2. Sample an impulse response g ~ N(0, K_eta)
    # Add a tiny jitter to the diagonal for numerical stability (Cholesky decomposition)
    jitter = torch.eye(g_length) * jitter_regularization_parameter

    # Create a Multivariate Normal distribution directly on the GPU/CPU and sample
    mvn = torch.distributions.MultivariateNormal(
        loc=torch.zeros(g_length), covariance_matrix=K_eta + jitter
    )
    g = mvn.sample()

    # 3. Generate input sequence u_t (e.g., white Gaussian noise)
    u = torch.randn(n)

    # 4. Compute noise-free output y^0 using convolution
    # To mimic scipy.signal.convolve(..., mode="full") in PyTorch:
    # - Reshape inputs to (batch, channel, length) -> (1, 1, L)
    # - PyTorch's F.conv1d computes cross-correlation, so we flip the filter 'g' to perform true convolution
    # - Add padding of 'g_length - 1' to get the "full" overlap
    u_reshaped = u.view(1, 1, n)
    g_flipped = torch.flip(g, dims=[0]).view(1, 1, g_length)

    y0_full = F.conv1d(u_reshaped, g_flipped, padding=g_length - 1)

    # Squeeze back to 1D and slice to length 'n'
    y0 = y0_full[0, 0, :n]

    # 5. Add measurement noise e_t
    # Let's target a specific Signal-to-Noise Ratio (SNR)
    snr_db = (snr_max - snr_min) * torch.rand(1).item() + snr_min
    snr_linear = 10 ** (snr_db / 10)

    # PyTorch var unbiased=False matches numpy's default np.var behavior (ddof=0)
    noise_variance = torch.var(y0, unbiased=False) / snr_linear
    e = torch.randn(n) * torch.sqrt(noise_variance)

    y = y0 + e

    # 6. Split the sequences into past and future ---
    u_past = u[:n_past]
    y_past = y[:n_past]

    u_future = u[n_past:]
    y_future = y[n_past:]

    # 7. Store and return the dictionary matching your complex format
    return {
        "u_past": u_past,
        "y_past": y_past,
        "u_future": u_future,
        "y_future": y_future,
        "g": g,
        "eta": {"c": c, "alpha": alpha, "snr_db": snr_db},
    }
