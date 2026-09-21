# Technical Design Document

## 1. Project Scope

This repository is a research-oriented PyTorch project focused on the application of transformer-like architectures to control system prediction tasks.

The architecture is modular by design, separating:

- experiment configuration,
- dataset loading and preprocessing,
- model composition and variant selection,
- training, validation, and checkpoint workflows,
- logging and experiment tracking.

The codebase enables:

- preprocessing of standard datasets such as CIFAR-10 and MNIST into PyTorch tensor;
- construction and training of models composed of embedding layers, residual blocks, SSM modules, and classification/decoder heads;
- evaluation of custom SSM implementations, including both sequential and Triton-parallel kernels;
- experiment tracking through Hydra and optional Weights & Biases integration;
- cluster-compatible execution via Slurm.

### Functional Requirements

TODO

### Non-functional Requirements

TODO

## 2. High-Level Workflow

The main entry point is `training_script.py`, which uses Hydra to load configuration from `configs/config.yaml` and experiment-specific YAML file.
The workflow is as follows:

1. The training entrypoint `training_script.py` loads Hydra configuration and sets up the runtime environment.
2. The script configures the random seeds for reproducibility (via SeedManager), initializes the Weights & Biases logger (WandDBLogger), determines the optimal hardware device (get_torch_device), and builds the PyTorch DataLoaders for the training, validation, and test splits.
3. Execution Mode Selection (Train, Resume, or Inference): The script checks the configuration for specific flags to dictate the run mode:

- Standard Training: If no special flags are set, it initializes a fresh model and optimizer based on the defined strategy and begins training from epoch 1.
- Resuming Training (resume_checkpoint): If a valid path is provided to resume_checkpoint in the configuration, the script loads the model weights, optimizer state, starting epoch, and previous best metrics. Important: When resuming training, you must ensure that the Hydra configuration used matches the exact configuration that created the checkpoint. Altering model architecture, dataset parameters, or strategy configurations will cause dimension mismatches or undefined behavior.
- Inference-Only Mode (inference=True): If the inference flag is set to True, the entire training process is skipped. This is designed for evaluating previously fully trained models.

4. Training Loop and Validation (if applicable): The model iterates through the training data, utilizing the customized strategy to inject specific logic. After each epoch, the model is evaluated on the validation set. The CheckpointManager monitors the validation metrics and automatically saves standard checkpoints, as well as distinct "best" models based on both the highest validation accuracy and the lowest validation loss. Early stopping is triggered if the target fit metric is reached.
5. Testing and Evaluation: Regardless of whether the script ran a full training loop or was launched in inference=True mode, it concludes by executing the test() function.
   The CheckpointManager loads the two optimal models: one prioritizing accuracy (best_model_acc) and the other prioritizing loss (best_model_loss).
   Important for Inference Mode: If you are running with inference=True, you must ensure that the pre-trained best model files are correctly placed in the corresponding Hydra output folder (base_dir) so the CheckpointManager can find and load them.
   The script evaluates both best models against the test dataset, computes baseline statistical estimates (LS and MMSE baselines), and logs the final comparative metrics via WandB before exiting. Optionally, it cleans up the saved models if save_model is set to false.
   computes loss, and updates weights.

Examples:

- python -m training_script experiment=dynamic-impulse-response-prediction
- python -m training_script experiment=dynamic-impulse-response-prediction resume_checkpoint="saved_models/my_model.pt"
- python -m training_script experiment=dynamic-impulse-response-prediction inference=True

## 3. Configuration and Experiment Management

The project uses Hydra for configuration management. The top-level `configs/config.yaml` defines common runtime settings such as:

- `num_workers`,
- `seed`,
- `debug_every`,
- dataset and model defaults.

Per-experiment configuration files live in `configs/experiment/` and select dataset/model variants through Hydra defaults.
See the specific experiment YAML file for examples of how to configure different datasets, model architectures, and training hyperparameters.
Hydra also controls the output directory layout and supports Slurm job integration through a custom resolver in `training_script.py`.

## 4. Project Structure

### Architectural Patterns

- Hydra-driven configuration for experiment definitions and output directory management.
- Factory patterns for model and dataset construction.
- PyTorch module composition for neural architectures.
- Separation of concerns between preprocessing, data loading, model definition, training, monitoring, and benchmarking.

### Main Directories

- `configs/` — Hydra configuration files and experiment presets.
- `datasets/` — dataset classes and data loader factory utilities.
- `models/` — PyTorch model implementations, SSM modules, and output layers.
- `outputs/`: default experiment outputs saved by Hydra.
- `preprocessing/` — data preparation scripts.
- `utils/` — environment helpers, directory constants, and metrics utilities.
- `monitor/` — experiment tracking and logging utilities.
- `benchmarks/` — benchmarking scripts, plotting utilities, kernel experiments.
- `slurm/` — batch submission scripts for cluster execution.
- `training_script.py` — main entry point for training experiments.

## 6. Model Architecture

TODO

## 7. Design Intent

TODO
