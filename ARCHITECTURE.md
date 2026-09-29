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

## 2. High-Level Workflow

The main entry point is `training_script.py`, which uses Hydra to load configuration from `configs/config.yaml` and experiment-specific YAML files.

The workflow is as follows:

1. **Environment Setup:** The entry point loads the Hydra configuration, seeds the random number generators for reproducibility (via `SeedManager`), initializes the Weights & Biases logger (`WandDBLogger`), determines the optimal hardware device (`get_torch_device`), and builds the PyTorch DataLoaders for the training, validation, and test splits.
2. **Model Initialization (`model_mode`):** The script checks the `model_mode` parameter to determine how the model weights are initialized:
   - **`new`:** Initializes a fresh model from scratch based on the defined architecture and strategy.
   - **`load`:** Initializes the model and loads pre-trained weights from the exact file path specified by the `model_path` parameter via the `CheckpointManager`.
3. **Execution Mode Selection (`running_mode`):** The script dictates the execution flow based on the `running_mode` parameter:
   - **`training`:** Executes the full training loop.
     - _Resuming:_ If a valid `resume_checkpoint` path is provided in the config, it restores the optimizer state, starting epoch, RNG states, and previous best metrics before resuming the loop.
     - _Validation & Checkpointing:_ After each epoch, the model is evaluated on the validation set. The `CheckpointManager` monitors metrics and saves standard checkpoints, alongside distinct "best" models based on both the highest validation accuracy and the lowest validation loss. Early stopping triggers if the target FIT metric is reached.
     - _Testing:_ Once training concludes, it automatically loads the "best" models and runs the testing and baseline evaluation phase.
   - **`inference`:** Skips training entirely. It evaluates the current model directly on the test dataset. (Typically used in conjunction with `model_mode=load`).
   - **`diagnostic`:** Executes a diagnostic evaluation across the Training, Validation, and Test sets sequentially. It computes the FIT for each split to verify whether the model is exhibiting "benign interpolation" (memorizing the finite training sample) or acting as a population Bayes estimator.
4. **Testing and Baseline Evaluation:** During `training` or `inference` modes, the script concludes by running the `test()` function. It evaluates the model against the test dataset, computes baseline statistical estimates (Least Squares and MMSE baselines), and logs the final comparative metrics via WandB before exiting. Optionally, it cleans up the saved models if `save_model` is set to false.

**Execution Examples:**

**Standard Training (Fresh Model):**

```bash
python -m training_script running_mode=training model_mode=new experiment=dynamic-impulse-response-prediction
```

**Resume Interrupted Training:**

```bash
python -m training_script running_mode=training model_mode=new resume_checkpoint="outputs/checkpoints/model.pt" experiment=dynamic-impulse-response-prediction
```

**Inference-Only (Evaluate a Pre-trained Model):**

```bash
python -m training_script running_mode=inference model_mode=load model_path="'/path/to/best_model.pt'" experiment=dynamic-impulse-response-prediction
```

**Diagnostic Evaluation (Train/Val/Test FIT checks):**

```bash
python -m training_script running_mode=diagnostic model_mode=load model_path="'saved_models/best_model.pt'" experiment=dynamic-impulse-response-prediction-snr3-10-4-64-16
```

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
