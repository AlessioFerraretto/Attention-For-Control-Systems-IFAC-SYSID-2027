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

        # Updates best_fit by saving the maximum value element by element in the batch
        best_fit = torch.maximum(best_fit, fit_metric)

    # Returns the tensor of best fits for the entire batch
    return None, best_fit


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

    return None, fit_metric
