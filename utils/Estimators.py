import logging

import torch

log = logging.getLogger("script")


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
    model,
    inputs_orig,
    masks,
    u,
    y,
    noise_var,
    g_length,
    task,
    M=5,
):
    """
    Computes the effective Degrees of Freedom (df) and SURE estimate using Hutchinson's stochastic trace method.
    Relies on Task polymorphism to route gradient injection and y_hat construction.
    """
    B, n = y.shape

    y_with_grad = y.clone().requires_grad_(True)
    inputs_with_grad = inputs_orig.clone()

    # The task explicitly defines if it needs 'y' in the input tensor
    if task.requires_y_in_input:
        inputs_with_grad[:, :n, 1] = y_with_grad

    # 1. Forward pass
    preds, _ = model(inputs_with_grad, masks)

    # 2. Polymorphic y_hat construction
    y_hat = task.compute_y_hat(
        preds=preds,
        u=u,
        y_with_grad=y_with_grad,
        noise_var=noise_var,
        g_length=g_length,
    )

    # 3. Hutchinson Trace Estimation
    df_estimate = torch.zeros(B, device=y.device)
    for _ in range(M):
        z = (torch.randint(0, 2, size=y_hat.shape, device=y.device).float() * 2) - 1.0

        vjp = torch.autograd.grad(
            outputs=y_hat,
            inputs=y_with_grad,
            grad_outputs=z,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]

        if vjp is not None:
            df_estimate += torch.sum(z * vjp, dim=1)

    df_estimate /= M

    # 4. Compute SURE Risk
    residuals_sq = torch.sum((y - y_hat) ** 2, dim=1)
    sure_estimate = (
        residuals_sq + 2 * noise_var.squeeze() * df_estimate - n * noise_var.squeeze()
    )

    return df_estimate.detach(), sure_estimate.detach()


def compute_mmse_df_and_sure(u, y, noise_var, K, g_length):
    """
    Computes the exact Degrees of Freedom (df) and SURE estimate for the Bayesian MMSE estimator.
    """
    B, n = y.shape

    # 1. Build the Toeplitz matrix U for the batch
    u_padded = torch.nn.functional.pad(u, (g_length - 1, 0))
    U = u_padded.unfold(dimension=1, size=g_length, step=1)
    U = torch.flip(U, dims=[2])  # Shape: (B, n, g_length)
    U_T = U.transpose(1, 2)  # Shape: (B, g_length, n)

    # 2. Compute U K U'
    UK = torch.bmm(U, K)
    UKUT = torch.bmm(UK, U_T)  # Shape: (B, n, n)

    # 3. Add the noise variance to the diagonal (sigma^2 I)
    I = torch.eye(n, device=y.device).unsqueeze(0).expand(B, n, n)
    sigma2_I = noise_var.view(B, 1, 1) * I

    # 4. Solve the system to obtain the projection matrix H = U K U' (U K U' + sigma^2 I)^(-1)
    # Use torch.linalg.solve for numerical stability (solve AX = B).
    # Since the matrices are symmetric, this is equivalent to computing the inverse.
    H = torch.linalg.solve(UKUT + sigma2_I, UKUT)

    # 5. The exact degrees of freedom are the trace of matrix H
    df_estimate = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)

    # 6. Compute y_hat and SURE
    # y_hat = H y
    y_hat = torch.bmm(H, y.unsqueeze(-1)).squeeze(-1)
    residuals_sq = torch.sum((y - y_hat) ** 2, dim=1)

    sure_estimate = (
        residuals_sq + 2 * noise_var.squeeze() * df_estimate - n * noise_var.squeeze()
    )

    return df_estimate.detach(), sure_estimate.detach()


def compute_ls_df_and_sure(u, y, noise_var, ls_estimate, best_order):
    """
    Computes the Degrees of Freedom (df) and SURE estimate for the Least Squares estimator.
    For LS, the degrees of freedom exactly match the chosen model order.
    """
    B, n = y.shape
    g_length = ls_estimate.shape[-1]

    # 1. The degrees of freedom for LS match the number of estimated parameters (the order)
    df_estimate = best_order.float()

    # 2. Calculate y_hat = U * g_hat using the Toeplitz matrix approach
    u_padded = torch.nn.functional.pad(u, (g_length - 1, 0))
    U = u_padded.unfold(dimension=1, size=g_length, step=1)
    U = torch.flip(U, dims=[2])

    y_hat = torch.bmm(U, ls_estimate.unsqueeze(-1)).squeeze(-1)  # Shape: (B, n)

    # 3. Compute SURE Risk estimate: ||y - y_hat||^2 + 2 * \sigma^2 * df - n * \sigma^2
    residuals_sq = torch.sum((y - y_hat) ** 2, dim=1)
    sure_estimate = (
        residuals_sq + 2 * noise_var.squeeze() * df_estimate - n * noise_var.squeeze()
    )

    return df_estimate.detach(), sure_estimate.detach()


def compute_impulseest(
    input_batch: torch.Tensor,
    output_batch: torch.Tensor,
    true_impulse_batch: torch.Tensor,
    noise_var_batch: torch.Tensor,
    matrix_kernel_batch: torch.Tensor,
):
    """
    Estimates the impulse response using the TC (Tuned and Correlated) kernel.
    This is mathematically equivalent to the MMSE estimator but solved via
    the regularized normal equations (Primal space) instead of the Dual space.
    """
    B, n = input_batch.shape
    g_length = matrix_kernel_batch.shape[-1]
    device = input_batch.device
    dtype = input_batch.dtype

    # 1. Build the Toeplitz matrix U
    u_padded = torch.nn.functional.pad(input_batch, (g_length - 1, 0))
    U = u_padded.unfold(dimension=1, size=g_length, step=1)
    U = torch.flip(U, dims=[2])  # Shape: (B, n, g_length)
    U_T = U.transpose(1, 2)  # Shape: (B, g_length, n)

    # 2. Compute U^T * U
    UT_U = torch.bmm(U_T, U)  # Shape: (B, g_length, g_length)

    # 3. Formulate the TC Kernel regularized system: (K * U^T * U + sigma^2 * I) g = K * U^T * y
    # We multiply by K to avoid computing K^-1, which is numerically unstable.
    K_UT_U = torch.bmm(matrix_kernel_batch, UT_U)

    I = torch.eye(g_length, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
    sigma2_I = noise_var_batch.view(B, 1, 1) * I

    # Left hand side of the equation (A)
    LHS = K_UT_U + sigma2_I

    # Right hand side of the equation (b)
    UT_y = torch.bmm(U_T, output_batch.unsqueeze(-1))
    RHS = torch.bmm(matrix_kernel_batch, UT_y)

    # 4. Solve the linear system
    g_hat = torch.linalg.solve(LHS, RHS).squeeze(-1)

    # 5. Calculate FIT
    norm_difference = torch.linalg.vector_norm(true_impulse_batch - g_hat, dim=-1)
    norm_true_response = torch.linalg.vector_norm(true_impulse_batch, dim=-1)
    fit_metric = 100.0 * (1.0 - (norm_difference / (norm_true_response + 1e-8)))

    return g_hat, fit_metric


def compute_impulseest_df_and_sure(
    u: torch.Tensor,
    y: torch.Tensor,
    noise_var_batch: torch.Tensor,
    matrix_kernel_batch: torch.Tensor,
    g_length: int,
):
    """
    Computes the Degrees of Freedom (df) and SURE estimate for the TC kernel estimator.
    It uses the primal space formulation to match compute_impulseest_tc.
    """
    B, n = y.shape
    device = y.device
    dtype = y.dtype

    # 1. Build the Toeplitz matrix U
    u_padded = torch.nn.functional.pad(u, (g_length - 1, 0))
    U = u_padded.unfold(dimension=1, size=g_length, step=1)
    U = torch.flip(U, dims=[2])
    U_T = U.transpose(1, 2)

    # 2. Build K * U^T * U
    UT_U = torch.bmm(U_T, U)
    K_UT_U = torch.bmm(matrix_kernel_batch, UT_U)

    # 3. Add noise variance: K * U^T * U + sigma^2 * I
    I = torch.eye(g_length, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
    sigma2_I = noise_var_batch.view(B, 1, 1) * I
    LHS = K_UT_U + sigma2_I

    # 4. Compute projection matrix H = U * (K * U^T * U + sigma^2 * I)^(-1) * K * U^T
    # First, solve LHS * X = K * U^T  --> X will be (K * U^T * U + sigma^2 * I)^(-1) * K * U^T
    K_UT = torch.bmm(matrix_kernel_batch, U_T)
    X = torch.linalg.solve(LHS, K_UT)

    H = torch.bmm(U, X)  # Shape: (B, n, n)

    # 5. The exact degrees of freedom are the trace of the projection matrix H
    df_estimate = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)

    # 6. Compute y_hat = H * y and SURE
    y_hat = torch.bmm(H, y.unsqueeze(-1)).squeeze(-1)
    residuals_sq = torch.sum((y - y_hat) ** 2, dim=1)

    sure_estimate = (
        residuals_sq
        + 2 * noise_var_batch.squeeze() * df_estimate
        - n * noise_var_batch.squeeze()
    )

    return df_estimate.detach(), sure_estimate.detach()


def verify_sure_ranking(
    models,
    avg_sure,
    avg_fit,
    avg_fit_mmse,
    avg_sure_mmse,
    avg_fit_ls,
    avg_sure_ls,
    avg_fit_impulseest,
    avg_sure_impulseest,
):
    # --- VERIFY SURE RANKING VS TRUE FIT RANKING ---
    sure_values = {}
    fit_values = {}

    # 1. Add Transformer models
    for crit in models:
        crit_name = crit.name if hasattr(crit, "name") else str(crit).upper()
        sure_values[f"Transformer_{crit_name}"] = avg_sure[crit]
        fit_values[f"Transformer_{crit_name}"] = avg_fit[crit]

    # 2. Add MMSE baseline (competing estimator)
    if avg_sure_mmse is not None and avg_fit_mmse is not None:
        sure_values["MMSE"] = avg_sure_mmse
        fit_values["MMSE"] = avg_fit_mmse

    # 2. Add LS baseline
    if avg_sure_ls is not None and avg_fit_ls is not None:
        sure_values["LS"] = avg_sure_ls
        fit_values["LS"] = avg_fit_ls

    # 2. Add Impulseest baseline
    if avg_sure_impulseest is not None and avg_fit_impulseest is not None:
        sure_values["Impulseest"] = avg_sure_impulseest
        fit_values["Impulseest"] = avg_fit_impulseest

    # 3. Sort rankings
    # SURE: ascending order (lowest risk is best)
    ranked_by_sure = sorted(sure_values.keys(), key=lambda k: sure_values[k])

    # FIT: descending order (highest precision is best)
    ranked_by_fit = sorted(fit_values.keys(), key=lambda k: fit_values[k], reverse=True)

    print("\n" + "-" * 90)
    print(" " * 20 + "SURE RANKING vs TRUE FIT RANKING")
    print("-" * 90)
    log.info(f"Ranking by SURE (lowest risk best): {ranked_by_sure}")
    log.info(f"Ranking by TRUE FIT (highest best): {ranked_by_fit}")
    print("-" * 90 + "\n")
