import matplotlib.pyplot as plt
import numpy as np
import torch


def evaluate_model(
    model,
    cfg,
    dataset,  # The dataset object passed from the training loop
    device,
    num_test_samples=3,
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
            inputs_tensor, target_tensor = dataset[i]

            # Add batch dimension and move to device for inference
            inputs = inputs_tensor.unsqueeze(0).to(device)

            # Inference
            preds, logits = model(inputs)
            y_hat_future = preds[0, -n_future:].squeeze().cpu().numpy()

            # --- Extract data for plotting directly from the tensor ---
            inputs_np = inputs_tensor.cpu().numpy()
            target_np = target_tensor.cpu().numpy()

            # Remove any size-1 dimensions (e.g. turns (200, 1) into (200,))
            inputs_np = inputs_np.squeeze()
            y_future_true = target_np.squeeze()

            # Safely extract based on the shape of the tensor
            if inputs_np.ndim == 2 and inputs_np.shape[1] >= 2:
                # Format: [100, 2] (u and y are parallel features)
                u_past = inputs_np[:n_past, 0]
                u_future = inputs_np[n_past:, 0]
                y_past = inputs_np[:n_past, 1]
            elif inputs_np.ndim == 1 and len(inputs_np) >= (n_past * 2):
                # Format: [200] (Concatenated sequentially)
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
    plt.show()
