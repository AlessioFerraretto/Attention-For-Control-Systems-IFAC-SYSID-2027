import itertools
import os

from omegaconf import OmegaConf

# 1. Define the parameter grids
VAR_1 = [2, 4, 6, 8]  # num_layers
VAR_2 = [64, 128, 256]  # model_dim
VAR_3 = [16, 32]  # num_heads

output_dir = "configs/experiment/bulk/"
os.makedirs(output_dir, exist_ok=True)

combinations = list(itertools.product(VAR_1, VAR_2, VAR_3))
print(f"Generating {len(combinations)} configuration files...")
input("Press Enter to continue or Ctrl+C to abort...")

# Iterate over each combination
for v1, v2, v3 in combinations:
    # 2. Load the base config fresh for each iteration
    # Ensure you have a 'base_config.yaml' file in your directory
    config = OmegaConf.load(
        "configs/experiment/dynamic-impulse-response-prediction.yaml"
    )

    # 3. Override the specific variables
    config.running_mode = "training"
    config.model_mode = "new"
    # config.model_mode = "load"
    # config.model_path = f"/home/ferraretto/Desktop/Attention For Control Systems/saved_models/model_{v1}_{v2}_{v3}.pt"

    config.transformer_seed = 1
    config.data_seed = 1
    config.val_seed = 2
    config.test_seed = 3

    config.epochs = 1000
    config.eval_every = 1000
    config.learning_rate = 1e-3

    config.training_batch_size = 64
    config.validation_batch_size = 64
    config.test_batch_size = 1

    config.save_model = True
    config.save_checkpoints = False
    config.wandb.enable = False

    config.model.num_layers = v1
    config.model.model_dim = v2
    config.model.num_heads = v3

    # Set all dataset defaults
    config.dataset.kernel = "stable_spline"
    config.dataset.batches_per_epoch = 100
    config.dataset.g_length = 64
    config.dataset.n = 100

    config.dataset.c_min = 0.0001
    config.dataset.c_max = 1.0
    config.dataset.alpha_min = 0.70
    config.dataset.alpha_max = 0.95

    config.dataset.eta_min = 0.1
    config.dataset.eta_max = 0.9

    config.dataset.snr_min = 3
    config.dataset.snr_max = 10

    config.dataset.jitter_regularization_parameter = 0.00001

    config.dataset.filter_type = "type_1"
    config.dataset.a_param = 0.9

    # 4. Format the filename
    filename = f"{config.dataset.name}-{v1}-{v2}-{v3}.yaml"
    filepath = os.path.join(output_dir, filename)

    # 5. Save the new configuration
    with open(filepath, "w") as f:
        # OmegaConf strips comments when loading/saving.
        # We manually re-inject the Hydra global directive at the top.
        f.write("# @package _global_\n\n")
        OmegaConf.save(config, f)

print(f"Done! All files have been saved to the '{output_dir}' directory.")
