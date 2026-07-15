import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy import signal

from models.Factory import construct_model
from utils.Environment import get_job_id
from utils.StableSplineGenerator import generate_stable_spline

OmegaConf.register_new_resolver(name="slurm_job_id", resolver=get_job_id)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Configuration (must match training)
    future_horizon = 10
    num_test_samples = 3  # How many systems to visualize
    model_path = "model.pt"

    # 2. Set different seed for test data generation

    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = construct_model(cfg=cfg, device=device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Successfully loaded {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found.")
        return

    model.eval()

    # 4. Generate Test Data
    print("Generating test dataset...")
    samples = []
    for _ in range(num_test_samples):
        sample = generate_stable_spline(
            g_length=cfg.dataset.g_length,
            n_past=cfg.dataset.n_past,
            n_future=cfg.dataset.n_future,
            c_min=cfg.dataset.c_min,
            c_max=cfg.dataset.c_max,
            alpha_min=cfg.dataset.alpha_min,
            alpha_max=cfg.dataset.alpha_max,
            snr_min=cfg.dataset.snr_min,
            snr_max=cfg.dataset.snr_max,
            jitter_regularization_parameter=cfg.dataset.jitter_regularization_parameter,
        )
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
            u = np.concatenate([sample["u_past"], sample["u_future"]])
            y = np.concatenate([sample["y_past"], sample["y_future"]])
            inputs = (
                torch.tensor(np.stack([u, y], axis=1), dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )

            # Ground truth
            g_true = sample["g"]
            y_future_true = sample["y_future"]

            # Inference
            _, logits = model(inputs)
            g_hat = logits.squeeze(0).cpu().numpy()
            y_hat_future = signal.convolve(sample["u_future"], g_hat, mode="full")[
                : len(sample["u_future"])
            ]

            # --- Plot 1: Impulse Response (g) ---
            ax_g = axes[i, 0] if num_test_samples > 1 else axes[0]
            ax_g.plot(
                g_true, label="True $g$", marker="o", linestyle="-", color="black"
            )
            ax_g.plot(
                g_hat,
                label=r"Predicted $\hat{g}$",
                marker="x",
                linestyle="--",
                color="red",
            )
            ax_g.set_title(f"System {i + 1}: Impulse Response Estimation")
            ax_g.set_xlabel("Time Lag")
            ax_g.set_ylabel("Amplitude")
            ax_g.grid(True, alpha=0.3)
            ax_g.legend()

            # --- Plot 2: Future Output Prediction (y) ---
            ax_y = axes[i, 1] if num_test_samples > 1 else axes[1]
            time_steps = range(cfg.dataset.n_past, cfg.dataset.n_past + future_horizon)

            # Plot the last few past points for context
            past_context = 15
            context_steps = range(cfg.dataset.n_past - past_context, cfg.dataset.n_past)
            ax_y.plot(
                context_steps,
                sample["y_past"][-past_context:],
                color="gray",
                label="Past $y$",
            )

            future_time_steps = list(
                range(cfg.dataset.n_past, cfg.dataset.n_past + len(y_future_true))
            )
            ax_y.plot(
                future_time_steps,
                y_future_true,
                label="True Future $y$",
                marker="o",
                color="black",
            )
            ax_y.plot(
                future_time_steps,
                y_hat_future,
                label=r"Predicted Future $\hat{y}$",
                marker="x",
                linestyle="--",
                color="blue",
            )
            ax_y.axvline(
                x=cfg.dataset.n_past - 0.5,
                color="green",
                linestyle=":",
                label="Prediction Start",
            )

            ax_y.set_title(f"System {i + 1}: Future Output Prediction")
            ax_y.set_xlabel("Time Step")
            ax_y.set_ylabel("System Output")
            ax_y.grid(True, alpha=0.3)
            ax_y.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


if __name__ == "__main__":
    main()
