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
2. TODO
3. TODO
4. The training loop consumes batches, computes loss, and updates weights.
5. Validation uses held-out data to compute loss and exact-match metrics.
6. Checkpoints and best models are saved to Hydra-managed output locations.
7. Optional logging to Weights & Biases is performed if enabled in the configuration.

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
