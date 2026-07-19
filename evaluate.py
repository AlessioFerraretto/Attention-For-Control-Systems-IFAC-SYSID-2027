import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.StableSplineGenerator import simulate_io_data


def evaluate_model(model, cfg, device, g, n, snr_min, snr_max, num_test_samples=3):
    # 1. Configuration (must match training)
    future_horizon = cfg.dataset.n_future

    # 2. Set different seed for test data generation

    model.eval()

    # 4. Generate Test Data
    print("Generating test dataset...")
    samples = []
    n_past = cfg.dataset.n_past
    n_future = cfg.dataset.n_future
    for _ in range(num_test_samples):
        u, y, snr_db = simulate_io_data(g, n, snr_min, snr_max)
        # Split into past and future
        sample = {
            "u_past": u[:n_past].numpy(),
            "y_past": y[:n_past].numpy(),
            "u_future": u[n_past:].numpy(),
            "y_future": y[n_past:].numpy(),
            "g": g.numpy(),
        }
        samples.append(sample)

    # 5. Plotting Setup
    fig, axes = plt.subplots(
        nrows=num_test_samples, ncols=2, figsize=(14, 4 * num_test_samples)
    )
    fig.suptitle("Foundation Model System Identification Results", fontsize=16)

    with torch.no_grad():
        for i in range(num_test_samples):
            sample = samples[i]

            # Build the same input representation used during training.
            # Format: [u_past, y_past, u_future, padding]
            padding = np.zeros(cfg.dataset.n_future)
            input_sequence = np.concatenate(
                [
                    sample["u_past"],
                    sample["y_past"],
                    sample["u_future"],
                    padding,
                ]
            )
            inputs = (
                torch.tensor(input_sequence, dtype=torch.float32)
                .unsqueeze(-1)  # shape: (seq_len, 1)
                .unsqueeze(0)  # shape: (batch=1, seq_len, 1)
                .to(device)
            )

            # Ground truth
            y_future_true = sample["y_future"]

            # Inference
            preds, logits = model(inputs)
            y_hat_future = preds[0, -n_future:].squeeze().cpu().numpy()

            # --- Impostazione degli assi per il Plotting ---
            # Gestione sicura degli indici nel caso in cui num_test_samples sia 1
            if num_test_samples == 1:
                ax_y, ax_u = axes[0], axes[1]
            else:
                ax_y, ax_u = axes[i, 0], axes[i, 1]

            # Vettori temporali per asse X
            t_past = np.arange(n_past)
            t_future = np.arange(n_past, n_past + n_future)

            # --- Colonna 1: Plot dell'Output (y) Reale vs Predetto ---
            ax_y.plot(t_past, sample["y_past"], label="y_past (history)", color="gray")
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

            # --- Colonna 2: Plot dell'Input (u) di riferimento ---
            ax_u.plot(t_past, sample["u_past"], label="u_past", color="lightgray")
            ax_u.plot(t_future, sample["u_future"], label="u_future", color="green")

            ax_u.set_title(f"Sample {i + 1}: System Input (u)")
            ax_u.set_xlabel("Time step")
            ax_u.set_ylabel("Amplitude")
            ax_u.legend()
            ax_u.grid(True)

    # Regolazioni finali del layout del plot
    plt.tight_layout()
    # Lasciamo spazio in alto affinché il 'suptitle' non si sovrapponga ai grafici
    plt.subplots_adjust(top=0.92)
    plt.show()
