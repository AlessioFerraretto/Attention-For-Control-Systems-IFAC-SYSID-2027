"""
Modular training procedure that can manage custom training strategies
"""

import logging
import sys

# Windows compatibility: Python's default console encoding on Windows is often #(Nuova)
# cp1252, which cannot encode common Unicode characters (->, Greek letters,
# math symbols) that may appear in config descriptions, log messages, or
# docstrings. Reconfigure stdout/stderr to UTF-8 so that logging.StreamHandler
# can emit any Unicode without raising UnicodeEncodeError.
# This MUST be done before any other module configures logging.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass  # Best-effort; older Python or non-standard stream

from pathlib import Path
from typing import Literal

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from datasets.Factory import get_data_loaders
from models.Factory import construct_model
from monitor.WandDBLogger import WandDBLogger
from training_strategies.Factory import make_strategy
from utils.Environment import get_job_id, get_torch_device

log = logging.getLogger("script")

BestCriteria = Literal["acc", "loss"]

# Register the `get_job_id` as a custom OmegaConf resolver. This allows the
# output of the function to be accessed when building the Hydra configuration.
OmegaConf.register_new_resolver(name="slurm_job_id", resolver=get_job_id)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Entry point for the training script (Hydra-configured).

    Parameters:
        cfg : DictConfig
            Hydra configuration object for the experiment.

    Returns:
        None: Orchestrates training and testing workflow and persists artifacts.
    """

    wandb_logger = WandDBLogger(cfg)

    base_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"Results Directory: {base_dir}")

    device = get_torch_device()

    if cfg.seed:
        log.info(f"Fixing PRNG seed: {cfg.seed}")
        torch.manual_seed(seed=cfg.seed)

    train_loader, val_loader, test_loader = get_data_loaders(cfg=cfg, device=device)

    training(
        cfg=cfg,
        base_dir=base_dir,
        device=device,
        wandb_logger=wandb_logger,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    test(
        cfg=cfg,
        base_dir=base_dir,
        device=device,
        test_loader=test_loader,
        wandb_logger=wandb_logger,
    )


def training(
    cfg: DictConfig,
    base_dir: Path,
    device: torch.device,
    wandb_logger: WandDBLogger,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    """Train the configured model using provided data loaders.

    Parameters:
        cfg : DictConfig
            Experiment configuration containing hyperparameters and dataset/model specs.
        base_dir : Path
            Base output directory where checkpoints and artifacts are written.
        device : torch.device
            Torch device used for training (cuda/mps/cpu).
        wandb_logger : WandDBLogger
            Logger instance used to record metrics and artifacts.
        train_loader : DataLoader
            DataLoader providing training batches.
        val_loader : DataLoader
            DataLoader used to run quick validation checks each epoch.

    Returns:
        None: Checkpoints and best models are persisted to `base_dir`.
    """

    log.info(
        f"In this experiment we have '{cfg.model.description}' trained on '{cfg.dataset.description}'"
    )

    # ------------------------------------PREPARE DIRECTORIES------------------------------
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # ------------------------------------STRATEGY AND MODEL----------------------------
    model = construct_model(cfg=cfg, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Number of parameters: {total_params}")

    strategy = make_strategy(cfg)

    strategy.setup(model, cfg, device, train_loader)

    # ------------------------------LOSS, OPTIMIZER AND USEFUL VARIABLES--------------------
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = strategy.loss_fn()
    opt = strategy.make_optimizer(model, cfg)
    # Track the best validation accuracy seen during training.
    # Used to decide whether to save a new "best" model checkpoint.
    best_val_accuracy = 0.0

    # Track the best (lowest) validation loss.
    # Initialized to infinity so that any real loss will be lower.
    best_val_loss = float("inf")

    # Global step counter used for logging, scheduling, or checkpointing.
    # Increments every time an optimization step is performed.
    global_step = 0

    # ----------------------------TRAINING, VALIDATION PROCEDURE-----------------------------
    for epoch in range(1, cfg.epochs + 1):
        log.info(f"Epoch {epoch: 3d} / {cfg.epochs}")

        model.train(True)
        cumulative_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad()

            # Hook called before each training step.
            # This allows the strategy to update any dynamic parameters
            # (e.g. temperature τ, regularization λ, schedules, etc.)
            # based on the current global training step.
            strategy.before_step(model, global_step)

            output, _ = model(inputs)

            loss = loss_function(output, labels) + strategy.extra_loss(
                model, cfg, global_step
            )

            loss.backward()

            # Apply gradient clipping if the strategy defines it.
            # This helps stabilize training by preventing exploding gradients.
            strategy.clip_gradients(model)
            opt.step()
            # Advance the global training step counter.
            global_step += 1
            cumulative_loss += loss.item()

            if not (i - 1) % cfg.debug_every:
                wandb_dict, string_to_print = strategy.logging_metrics(model)
                log.info(
                    f"i: {i: 6d} | Loss: {loss.item():.6f}"
                )  # TODO: Check if it is correct: I removed "+ string_top_print"
                wandb_logger.log({"loss": loss.item()})
                wandb_logger.log(wandb_dict)

        log.info(f"Training Average Loss: {(cumulative_loss / len(train_loader)):.5f}")

        # ------------------------VALIDATION PHASE-----------------------------------------

        val_loss, val_accuracy = _evaluate(
            model, val_loader, loss_function, device, strategy=strategy
        )

        log.info(
            f"Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_accuracy:.3f}"
        )

        # Save checkpoint
        model_path = (
            checkpoints_dir / f"model_{epoch}_{val_loss:.5f}_{val_accuracy:.3f}.pt"
        )
        torch.save(
            obj={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": val_loss,
                "accuracy": val_accuracy,
            },
            f=model_path,
        )

        # Save best model in terms of accuracy (override the previous one)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_best_model(
                model,
                criteria="acc",
                base_dir=base_dir,
                opt=opt,
                loss=val_loss,
                accuracy=val_accuracy,
            )

        # Save best model in terms of loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_model(
                model,
                criteria="loss",
                base_dir=base_dir,
                opt=opt,
                loss=val_loss,
                accuracy=val_accuracy,
            )

        # ------------------------EARLY STOPPING-------------------------------
        # Once the task is essentially solved on the independent validation set,
        # there is no value in running the remaining epochs.
        early_stop_thr = cfg.get("early_stop_val_acc", 1.01)
        if val_accuracy >= early_stop_thr:
            log.info(
                f"Early stopping: validation accuracy {val_accuracy:.4f} "
                f">= threshold {early_stop_thr} at epoch {epoch}. "
                f"Skipping the remaining {cfg.epochs - epoch} epoch(s)."
            )
            break

    log.info("Training complete.")


def save_best_model(
    model: nn.Module,
    criteria: BestCriteria,
    base_dir: Path,
    opt: Optimizer,
    loss: float,
    accuracy: float,
):
    """Save the best model found so far according to the supplied criteria.

    Parameters:
        model : nn.Module
            Model whose state dict will be saved.
        criteria : BestCriteria
            Criterion identifier, either 'acc' or 'loss'.
        base_dir : Path
            Directory where the best model file will be saved.
        opt : Optimizer
            Optimizer whose state is saved alongside the model.
        loss : float
            Validation loss associated with this checkpoint.
        accuracy : float
            Validation accuracy associated with this checkpoint.

    Returns:
        None
    """
    for path in base_dir.glob(f"best_model_{criteria}*.pt"):
        log.debug(f"Removing previous best: {path}")
        path.unlink()

    model_path = base_dir / f"best_model_{criteria}_{loss:.3f}_{accuracy:.3f}.pt"
    torch.save(
        obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        },
        f=model_path,
    )


def _evaluate(model, loader, loss_fn, device, strategy=None):
    strategy.enter_eval(model) if strategy else None
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds, logits = model(inputs)
            total_loss += loss_fn(preds, labels).item() * inputs.size(0)
            # correct += torch.sum(preds == labels).item()
            performance_metrics = strategy.compute_metrics(preds, labels)
            correct += performance_metrics["correct_sequences"]
            total += inputs.size(0)
    strategy.exit_eval(model) if strategy else None
    return total_loss / max(total, 1), correct / max(total, 1)


def test(
    cfg: DictConfig,
    base_dir: Path,
    device: torch.device,
    test_loader: DataLoader,
    wandb_logger: WandDBLogger,
):
    """Evaluate best models (by accuracy and by loss) on the test set.

    Parameters:
        cfg : DictConfig
            Experiment configuration used to construct the model.
        base_dir : Path
            Directory containing the saved best model checkpoints.
        device : torch.device
            Device used for evaluation.
        test_loader : DataLoader
            DataLoader providing test batches.
        wandb_logger : WandDBLogger
            Logger instance for recording final test metrics.

    Returns:
        None: Logs test metrics and exits on missing checkpoints.
    """

    log.info(
        "Testing best models (found during training) in terms of loss and accuracy"
    )

    model_acc = construct_model(cfg=cfg, device=device)
    model_loss = construct_model(cfg=cfg, device=device)

    total_params = sum(p.numel() for p in model_acc.parameters())
    log.info(f"Number of parameters: {total_params}")

    acc_file = list(base_dir.glob("best_model_acc*.pt"))
    loss_file = list(base_dir.glob("best_model_loss*.pt"))

    if acc_file:
        best_acc = torch.load(f=acc_file[0], weights_only=False)
        log.info(f"Uploaded: {acc_file[0]}")
    else:
        log.error("Best accuracy model not found")
        sys.exit(1)

    if loss_file:
        best_loss = torch.load(f=loss_file[0], weights_only=False)
        log.info(f"Uploaded: {loss_file[0]}")
    else:
        log.error("Best loss model not found")
        sys.exit(1)

    model_acc.load_state_dict(best_acc["model_state_dict"])
    model_loss.load_state_dict(best_loss["model_state_dict"])

    # TODO: why is the loss function here instead of reading the one from the strategy?
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.MSELoss()  # TODO: Check if this is correct

    strategy_acc = make_strategy(cfg)
    strategy_acc.setup_for_eval(model_acc)
    test_loss_a, test_accuracy_a = _evaluate(
        model_acc, test_loader, loss_function, device, strategy=strategy_acc
    )

    strategy_loss = make_strategy(cfg)
    strategy_loss.setup_for_eval(model_loss)
    test_loss_l, test_accuracy_l = _evaluate(
        model_loss, test_loader, loss_function, device, strategy=strategy_loss
    )

    log.info(
        f"Best Model (by Accuracy) - Loss: {test_loss_a:.5f}, Accuracy: {test_accuracy_a:.3f}"
    )

    log.info(
        f"Best Model (by Loss) - Loss: {test_loss_l:.5f}, Accuracy: {test_accuracy_l:.3f}"
    )


if __name__ == "__main__":
    main()
