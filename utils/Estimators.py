import torch


def build_matrix_phi_batched(
    input_batch: torch.Tensor, horizon_length: int
) -> torch.Tensor:
    """Builds matrix_phi for an entire batch. input_batch has shape [batch_size, sequence_length]"""
    batch_size, sequence_length = input_batch.shape
    matrix_phi = torch.zeros(
        (batch_size, sequence_length, horizon_length),
        dtype=input_batch.dtype,
        device=input_batch.device,
    )

    for index in range(horizon_length):
        if index == 0:
            matrix_phi[:, :, index] = input_batch
        else:
            matrix_phi[:, index:, index] = input_batch[:, :-index]

    return matrix_phi


def compute_ls(
    input_batch: torch.Tensor,
    output_batch: torch.Tensor,
    true_impulse_batch: torch.Tensor,
    max_length: int,
):
    """Computes the Least Squares Oracle for the entire batch simultaneously."""
    batch_size, sequence_length = input_batch.shape
    best_fit = torch.full((batch_size,), float("-inf"), device=input_batch.device)
    best_order = torch.zeros(
        (batch_size,), dtype=torch.int64, device=input_batch.device
    )
    best_estimate = torch.zeros(
        (batch_size, max_length), dtype=input_batch.dtype, device=input_batch.device
    )

    # Add a dimension to the output for matrix calculation: [batch_size, sequence_length, 1]
    output_vector = output_batch.unsqueeze(-1)

    for horizon in range(1, max_length + 1):
        matrix_phi_batch = build_matrix_phi_batched(
            input_batch, horizon
        )  # [batch_size, sequence_length, horizon]

        # lstsq supports batches. Solves the system for all batch_size samples in parallel
        solution = torch.linalg.lstsq(
            matrix_phi_batch, output_vector
        ).solution  # [batch_size, horizon, 1]
        estimated_impulse_step = solution.squeeze(-1)  # [batch_size, horizon]

        # Padding
        padded_estimate = torch.zeros(
            (batch_size, max_length), dtype=input_batch.dtype, device=input_batch.device
        )
        padded_estimate[:, :horizon] = estimated_impulse_step

        # Vectorized FIT calculation along the sequence dimension (dim=-1)
        norm_difference = torch.linalg.vector_norm(
            true_impulse_batch - padded_estimate, dim=-1
        )
        norm_true_response = torch.linalg.vector_norm(true_impulse_batch, dim=-1)
        fit_metric = 100.0 * (1.0 - (norm_difference / (norm_true_response + 1e-8)))

        is_better = fit_metric > best_fit
        best_fit = torch.where(is_better, fit_metric, best_fit)
        best_estimate = torch.where(
            is_better.unsqueeze(-1), padded_estimate, best_estimate
        )
        best_order = torch.where(
            is_better, torch.full_like(best_order, horizon), best_order
        )

    return best_estimate, best_fit, best_order


def compute_mmse(
    input_batch: torch.Tensor,
    output_batch: torch.Tensor,
    true_impulse_batch: torch.Tensor,
    noise_variance_batch: torch.Tensor,
    matrix_kernel_batch: torch.Tensor,
):
    """Computes the Minimum Mean Square Error (MMSE) for the entire batch simultaneously."""
    batch_size, sequence_length = input_batch.shape
    impulse_length = matrix_kernel_batch.shape[-1]

    matrix_phi = build_matrix_phi_batched(
        input_batch, impulse_length
    )  # [batch_size, sequence_length, impulse_length]

    # Batched matrix multiplication (@ automatically handles batches in PyTorch)
    # transpose(1, 2) transposes only the matrix dimensions, leaving batch_size intact
    matrix_phi_kernel_phi_transposed = (
        matrix_phi @ matrix_kernel_batch @ matrix_phi.transpose(1, 2)
    )  # [batch_size, sequence_length, sequence_length]

    # Create the batched identity matrix multiplied by the variances
    # noise_variance_batch is [batch_size], we reshape it to [batch_size, 1, 1] for broadcasting
    matrix_identity = (
        torch.eye(sequence_length, dtype=input_batch.dtype, device=input_batch.device)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )
    matrix_noise_covariance = (
        noise_variance_batch.view(batch_size, 1, 1) * matrix_identity
    )

    # Matrix inversion on the entire batch
    matrix_inverse_term = torch.linalg.inv(
        matrix_phi_kernel_phi_transposed + matrix_noise_covariance
    )  # [batch_size, sequence_length, sequence_length]

    output_vector = output_batch.unsqueeze(-1)  # [batch_size, sequence_length, 1]

    estimated_impulse = (
        matrix_kernel_batch
        @ matrix_phi.transpose(1, 2)
        @ matrix_inverse_term
        @ output_vector
    )  # [batch_size, impulse_length, 1]
    estimated_impulse = estimated_impulse.squeeze(-1)  # [batch_size, impulse_length]

    # Vectorized FIT calculation
    norm_difference = torch.linalg.vector_norm(
        true_impulse_batch - estimated_impulse, dim=-1
    )
    norm_true_response = torch.linalg.vector_norm(true_impulse_batch, dim=-1)
    fit_metric = 100.0 * (1.0 - (norm_difference / (norm_true_response + 1e-8)))

    return estimated_impulse, fit_metric


def compute_hutchinson_df_and_sure(
    model, inputs_orig, masks, u, y, noise_var, g_length, M=5
):
    """
    Computes the effective Degrees of Freedom (df) and SURE estimate using Hutchinson's stochastic trace method.
    Addresses Hypothesis/Point 3 from Pillonetto's diagnostic notes.
    """
    B, n = y.shape
    device = y.device

    # 1. Prepare y for gradient computation
    y_with_grad = y.clone().requires_grad_(True)

    # 2. Reconstruct the input tensor while preserving the computational graph (based on the Dataset's __getitem__)
    uy = torch.stack([u, y_with_grad], dim=-1)  # (B, n, 2)
    padding_part = inputs_orig[:, n:, :]  # (B, g_length, 2)
    inputs_with_grad = torch.cat([uy, padding_part], dim=1)

    # 3. Forward pass of the "frozen" model
    g_hat, _ = model(inputs_with_grad, masks)

    # 4. Convolution y_hat = U * g_hat (Vectorized Toeplitz Matrix approach)
    # y_hat_i = sum_{j=0}^{g_length-1} g_j * u_{i-j}
    u_padded = torch.nn.functional.pad(u, (g_length - 1, 0))
    U = u_padded.unfold(dimension=1, size=g_length, step=1)
    U = torch.flip(U, dims=[2])
    y_hat = torch.bmm(U, g_hat.unsqueeze(-1)).squeeze(-1)  # Shape: (B, n)

    # 5. Hutchinson's method to estimate df = tr(dy_hat / dy)
    df_estimate = torch.zeros(B, device=device)
    for _ in range(M):
        # Rademacher vector (equiprobable -1 or +1 values)
        z = (torch.randint(0, 2, size=y_hat.shape, device=device).float() * 2) - 1.0

        # Vector-Jacobian Product (VJP) via autograd
        vjp = torch.autograd.grad(
            outputs=y_hat,
            inputs=y_with_grad,
            grad_outputs=z,
            retain_graph=True,
            create_graph=False,
        )[0]

        df_estimate += torch.sum(z * vjp, dim=1)

    df_estimate /= M

    # 6. Compute SURE Risk estimate: ||y - y_hat||^2 + 2 * \sigma^2 * df - n * \sigma^2
    residuals_sq = torch.sum((y - y_hat) ** 2, dim=1)
    sure_estimate = residuals_sq + 2 * noise_var * df_estimate - n * noise_var

    # Return detached estimates for safe logging and accumulation
    return df_estimate.detach(), sure_estimate.detach()
