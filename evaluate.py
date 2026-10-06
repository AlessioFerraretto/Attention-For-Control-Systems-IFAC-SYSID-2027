from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def evaluate_model(
    model,
    cfg,
    dataset,
    device,
    save_path,
    num_test_samples=3,
):
    if "impulse-response" in cfg.dataset.name:
        return evaluate_model_impulse_response_prediction(
            model, cfg, dataset, device, save_path, num_test_samples
        )
    elif "future" in cfg.dataset.name:
        return evaluate_model_future_prediction(
            model, cfg, dataset, device, save_path, num_test_samples
        )
    elif "kernel-parameters" in cfg.dataset.name:
        return evaluate_model_kernel_prediction(
            model, cfg, dataset, device, save_path, num_test_samples
        )
    else:
        raise ValueError(
            f"Unknown dataset name '{cfg.dataset.name}' for evaluation. Please check your configuration."
        )


def evaluate_model_kernel_prediction(
    model,
    cfg,
    dataset,
    device,
    save_path,
    num_test_samples,
):
    model.eval()

    # Grab the true dimensions from your config
    n = cfg.dataset.n

    # 1. Plotting Setup
    fig, axes = plt.subplots(
        nrows=num_test_samples, ncols=2, figsize=(14, 4 * num_test_samples)
    )
    fig.suptitle(
        "System Identification Results (Kernel Parameters: Alpha & c)", fontsize=16
    )

    with torch.no_grad():
        for i in range(num_test_samples):
            # Automatically generate and format data using your dataset's __getitem__
            (inputs, targets, info, *rest) = dataset[i]
            masks = rest[0].unsqueeze(0).to(device) if rest else None

            # Add batch dimension and move to device for inference
            inputs_tensor = inputs.unsqueeze(0).to(device)

            # Inference
            preds, *logits = model(inputs_tensor, mask=masks)

            # --- Extract data for plotting ---
            inputs_np = inputs_tensor.squeeze(0).cpu().numpy()

            # The model predicts exactly 2 values [alpha, c]. Flatten them to 1D arrays
            target_np = targets.cpu().numpy().reshape(-1)
            preds_np = preds.cpu().numpy().reshape(-1)

            # Extract u and y safely from the info dictionary
            # (since inputs for this task is the impulse response 'g', not u and y)
            if "u" in info and "y" in info:
                u_data = info["u"].cpu().numpy()[:n]
                y_data = info["y"].cpu().numpy()[:n]
            else:
                u_data = np.zeros(n)
                y_data = np.zeros(n)

            # --- Plotting Axes Setup ---
            if num_test_samples == 1:
                ax_uy, ax_params = axes[0], axes[1]
            else:
                ax_uy, ax_params = axes[i, 0], axes[i, 1]

            # Time axis for I/O plot
            t_data = np.arange(len(u_data))

            # --- Input/Output (u, y) Plot (Left side) ---
            ax_uy.plot(t_data, u_data, label="u (input)", color="lightgray")
            ax_uy.plot(t_data, y_data, label="y (output)", color="green", alpha=0.7)

            ax_uy.set_title(f"Sample {i + 1}: System I/O Trajectories")
            ax_uy.set_xlabel("Time step")
            ax_uy.set_ylabel("Amplitude")
            ax_uy.legend()
            ax_uy.grid(True)

            # --- Kernel Parameters Bar Chart (Right side) ---
            labels = ["Alpha", "c"]
            x = np.arange(len(labels))
            width = 0.35  # width of the bars

            # We assume target_np is [alpha, c]
            rects1 = ax_params.bar(
                x - width / 2, target_np[:2], width, label="True", color="blue"
            )
            rects2 = ax_params.bar(
                x + width / 2, preds_np[:2], width, label="Prediction", color="red"
            )

            ax_params.set_ylabel("Parameter Value")
            ax_params.set_title(f"Sample {i + 1}: Predicted vs True")
            ax_params.set_xticks(x)
            ax_params.set_xticklabels(labels)
            ax_params.legend()
            ax_params.grid(True, axis="y", linestyle="--", alpha=0.7)

            # Add numeric text annotations on top of the bars
            for j, (true_val, pred_val) in enumerate(zip(target_np[:2], preds_np[:2])):
                ax_params.text(
                    j - width / 2,
                    true_val,
                    f"{true_val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
                ax_params.text(
                    j + width / 2,
                    pred_val,
                    f"{pred_val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_model_impulse_response_prediction(
    model,
    cfg,
    dataset,
    device,
    save_path,
    num_test_samples,
):
    model.eval()

    # Grab the true dimensions from your config
    n = cfg.dataset.n
    g_length = cfg.dataset.g_length

    # 1. Plotting Setup
    fig, axes = plt.subplots(
        nrows=num_test_samples, ncols=2, figsize=(14, 4 * num_test_samples)
    )
    fig.suptitle("System Identification Results (Impulse Response)", fontsize=16)

    with torch.no_grad():
        for i in range(num_test_samples):
            # Automatically generate and format data using your dataset's __getitem__
            (inputs, targets, info, *rest) = dataset[i]
            masks = rest[0].unsqueeze(0).to(device) if rest else None

            # Add batch dimension and move to device for inference
            inputs = inputs.unsqueeze(0).to(device)

            # Inference
            preds, *logits = model(inputs, mask=masks)

            # --- Extract data for plotting ---
            inputs_np = inputs.squeeze(0).cpu().numpy()
            target_np = targets.cpu().numpy()

            # The model predicts the impulse response 'g' (length = g_length)
            # We use .reshape(-1) to ensure it stays a 1D array even if g_length is 1
            g_hat = preds.cpu().numpy().reshape(-1)
            g_true = target_np.reshape(-1)

            # Extract u and y from the inputs
            # We slice up to 'n' to drop the padding token at the end of the tensor
            u_data = inputs_np[:n, 0]
            y_data = inputs_np[:n, 1]

            # --- Plotting Axes Setup ---
            if num_test_samples == 1:
                ax_g, ax_uy = axes[0], axes[1]
            else:
                ax_g, ax_uy = axes[i, 0], axes[i, 1]

            # Time axes
            t_data = np.arange(n)
            t_g = np.arange(g_length)

            # --- Target (g) Plot ---
            ax_g.plot(t_g, g_true, label="g (ground truth)", color="blue", marker="o")
            ax_g.plot(
                t_g,
                g_hat,
                label="g (prediction)",
                color="red",
                linestyle="--",
                marker="x",
            )

            ax_g.set_title(f"Sample {i + 1}: Impulse Response (g)")
            ax_g.set_xlabel("Time step")
            ax_g.set_ylabel("Amplitude")
            ax_g.legend()
            ax_g.grid(True)

            # --- Input/Output (u, y) Plot ---
            ax_uy.plot(t_data, u_data, label="u (input)", color="lightgray")
            ax_uy.plot(t_data, y_data, label="y (output)", color="green", alpha=0.7)

            ax_uy.set_title(f"Sample {i + 1}: System I/O Trajectories")
            ax_uy.set_xlabel("Time step")
            ax_uy.set_ylabel("Amplitude")
            ax_uy.legend()
            ax_uy.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_model_future_prediction(
    model,
    cfg,
    dataset,
    device,
    save_path,
    num_test_samples,
):
    model.eval()
    n_past = cfg.dataset.n_past
    n_future = cfg.dataset.n_future

    # 1. Plotting Setup
    fig, axes = plt.subplots(
        nrows=num_test_samples, ncols=2, figsize=(14, 4 * num_test_samples)
    )
    fig.suptitle("Foundation Model System Identification Results", fontsize=16)

    with torch.no_grad():
        for i in range(num_test_samples):
            # Automatically generate and format data using your dataset's __getitem__
            (inputs, targets, info, *rest) = dataset[i]
            masks = rest[0].unsqueeze(0).to(device) if rest else None

            # Add batch dimension and move to device for inference
            inputs = inputs.unsqueeze(0).to(device)
            # Inference
            preds, logits = model(inputs, masks)
            y_hat_future = preds[0, -n_future:].squeeze().cpu().numpy()

            # --- Extract data for plotting directly from the tensor ---
            inputs_np = inputs.cpu().numpy()
            target_np = targets.cpu().numpy()

            # Remove any size-1 dimensions
            inputs_np = inputs_np.squeeze()
            y_future_true = target_np.squeeze()

            # Safely extract based on the shape of the tensor
            if inputs_np.ndim == 2 and inputs_np.shape[1] >= 2:
                # Format: [100, 2] (u and y are parallel features)
                u_past = inputs_np[:n_past, 0]
                u_future = inputs_np[n_past:, 0]
                y_past = inputs_np[:n_past, 1]
            elif inputs_np.ndim == 1 and len(inputs_np) >= (n_past * 2):
                u_past = inputs_np[:n_past]
                y_past = inputs_np[n_past : n_past * 2]
                u_future = inputs_np[n_past * 2 : n_past * 2 + n_future]
            else:
                # Fallback if shape is completely unexpected
                u_past = np.zeros(n_past)
                u_future = np.zeros(n_future)
                y_past = np.zeros(n_past)

            # --- Plotting Axes Setup ---
            if num_test_samples == 1:
                ax_y, ax_u = axes[0], axes[1]
            else:
                ax_y, ax_u = axes[i, 0], axes[i, 1]

            t_past = np.arange(n_past)
            t_future = np.arange(n_past, n_past + n_future)

            # --- Output (y) Plot ---
            ax_y.plot(t_past, y_past, label="y_past (history)", color="gray")
            ax_y.plot(
                t_future, y_future_true, label="y_future (ground truth)", color="blue"
            )
            ax_y.plot(
                t_future,
                y_hat_future,
                label="y_future (prediction)",
                color="red",
                linestyle="--",
            )

            ax_y.set_title(f"Sample {i + 1}: System Output (y)")
            ax_y.set_xlabel("Time step")
            ax_y.set_ylabel("Amplitude")
            ax_y.legend()
            ax_y.grid(True)

            # --- Input (u) Plot ---
            ax_u.plot(t_past, u_past, label="u_past", color="lightgray")
            ax_u.plot(t_future, u_future, label="u_future", color="green")

            ax_u.set_title(f"Sample {i + 1}: System Input (u)")
            ax_u.set_xlabel("Time step")
            ax_u.set_ylabel("Amplitude")
            ax_u.legend()
            ax_u.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
