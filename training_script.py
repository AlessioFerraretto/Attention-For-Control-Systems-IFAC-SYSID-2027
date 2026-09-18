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
from evaluate import evaluate_model
from models.Factory import construct_model
from monitor.WandDBLogger import WandDBLogger
from training_strategies.Factory import make_strategy
from utils.Environment import get_job_id, get_torch_device
from utils.Estimators import compute_ls, compute_mmse
from utils.SeedManager import SeedManager

log = logging.getLogger("script")

BestCriteria = Literal["acc", "loss"]

# Register the `get_job_id` as a custom OmegaConf resolver. This allows the
# output of the function to be accessed when building the Hydra configuration.
OmegaConf.register_new_resolver(name="slurm_job_id", resolver=get_job_id)
OmegaConf.register_new_resolver("mul", lambda x, y: int(x * y))
OmegaConf.register_new_resolver("div", lambda x, y: int(x / y))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Entry point for the training script (Hydra-configured).

    Parameters:
        cfg : DictConfig
            Hydra configuration object for the experiment.

    Returns:
        None: Orchestrates training and testing workflow and persists artifacts.
    """

    seed_manager = SeedManager()
    seed_manager.seed_dataset(cfg.data_seed)
    seed_manager.seed_transformer(cfg.transformer_seed)

    wandb_logger = WandDBLogger(cfg)

    base_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"Results Directory: {base_dir}")

    device = get_torch_device()

    train_loader, val_loader, test_loader = get_data_loaders(cfg=cfg)

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
    best_fit = float("-inf")

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
        for i, (inputs, targets, info, *rest) in enumerate(train_loader, start=1):
            masks = rest[0].to(device) if rest else None
            inputs, targets = (
                inputs.to(device),
                targets.to(device),
            )

            opt.zero_grad()

            # Hook called before each training step.
            # This allows the strategy to update any dynamic parameters
            # (e.g. temperature τ, regularization λ, schedules, etc.)
            # based on the current global training step.
            strategy.before_step(model, global_step)

            output, _ = model(inputs, masks)

            loss = loss_function(output, targets) + strategy.extra_loss(
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

        val_loss, fit = _evaluate(
            model,
            val_loader,
            loss_function,
            device,
            strategy=strategy,
            wandb_logger=wandb_logger,
        )

        log.info(f"Validation Loss: {val_loss:.5f}, FIT: {fit:.3f}")

        # ------------------------VISUALIZATION PHASE-----------------------------------
        if epoch % cfg.eval_every == 0:
            log.info(f"Running model evaluation at epoch {epoch}...")
            try:
                evaluate_model(
                    model=model,
                    cfg=cfg,
                    dataset=val_loader.dataset,
                    device=device,
                    save_path=base_dir / "images" / f"epoch_{epoch}.png",
                    num_test_samples=3,
                )
            except Exception as e:
                log.error(f"Evaluation failed: {e}")

        # ------------------------EARLY STOPPING-------------------------------
        # Once the task is essentially solved on the independent validation set,
        # there is no value in running the remaining epochs.
        if fit >= 100:
            log.info(
                f"Early stopping"
                f">= threshold at epoch {epoch}. "
                f"Skipping the remaining {cfg.epochs - epoch} epoch(s)."
            )
            break

        # ------------------------SAVE PHASE-----------------------------------------

        # Save checkpoint
        if cfg.save_checkpoints:
            model_path = checkpoints_dir / f"model_{epoch}_{val_loss:.5f}_{fit:.3f}.pt"
            torch.save(
                obj={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": val_loss,
                    "accuracy": fit,
                },
                f=model_path,
            )

        # Save best model in terms of accuracy (override the previous one)
        if fit > best_fit:
            best_fit = fit
            save_best_model(
                model,
                criteria="acc",
                base_dir=base_dir,
                opt=opt,
                loss=val_loss,
                accuracy=fit,
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
                accuracy=fit,
            )

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


def compute_ls_mmse(
    test_loader: DataLoader,
    device: torch.device,
    model_acc: nn.Module,
    model_loss: nn.Module,
    base_dir: Path,
):

    total = 0
    total_fit_ls = 0.0
    total_fit_mmse = 0.0
    to_save = {
        "true_impulse_response": [],
        "ls_estimate": [],
        "mmse_estimate": [],
        "model_best_accuracy": [],
        "model_best_loss": [],
        "best_order": [],
    }

    model_acc.eval()
    model_loss.eval()
    with torch.no_grad():
        for inputs, targets, info, *rest in test_loader:
            batch_size = inputs.size(0)

            u_batch = info["u"].to(device)
            y_batch = info["y"].to(device)
            noise_var_batch = info["noise_variance"].to(device)
            kernel_matrix_batch = info["kernel_matrix"].to(device)
            g_true_batch = targets.to(device)
            masks = rest[0].to(device) if rest else None

            ls_estimate, fit_ls_batch, best_order = compute_ls(
                u_batch, y_batch, g_true_batch, test_loader.dataset.g_length
            )
            mmse_estimate, fit_mmse_batch = compute_mmse(
                u_batch, y_batch, g_true_batch, noise_var_batch, kernel_matrix_batch
            )
            model_acc_estimate, _ = model_acc(inputs.to(device), masks)
            model_loss_estimate, _ = model_loss(inputs.to(device), masks)

            total_fit_ls += fit_ls_batch.sum().item()
            total_fit_mmse += fit_mmse_batch.sum().item()
            total += batch_size

            to_save["true_impulse_response"].append(g_true_batch.cpu())
            to_save["ls_estimate"].append(ls_estimate.cpu())
            to_save["mmse_estimate"].append(mmse_estimate.cpu())
            to_save["model_best_accuracy"].append(model_acc_estimate.cpu())
            to_save["model_best_loss"].append(model_loss_estimate.cpu())
            to_save["best_order"].append(best_order.cpu())

    average_fit_ls = total_fit_ls / max(total, 1)
    average_fit_mmse = total_fit_mmse / max(total, 1)

    log.info("Evaluation LS and MMSE on the test set")
    log.info(f"LS FIT: {average_fit_ls:.3f}")
    log.info(f"MMSE FIT: {average_fit_mmse:.3f}")

    torch.save(
        {key: torch.cat(value) for key, value in to_save.items()},
        base_dir / "test_impulse_responses.pt",
    )

    return average_fit_ls, average_fit_mmse


def _evaluate(model, loader, loss_fn, device, strategy=None, wandb_logger=None):
    strategy.enter_eval(model) if strategy else None
    model.eval()

    total_loss = 0.0
    fit = 0
    total = 0

    with torch.no_grad():
        for inputs, targets, info, *rest in loader:
            masks = rest[0].to(device) if rest else None
            inputs, targets = (
                inputs.to(device),
                targets.to(device),
            )

            preds, logits = model(inputs, masks)
            batch_size = inputs.size(0)

            total_loss += loss_fn(preds, targets).item() * batch_size

            performance_metrics = strategy.compute_metrics(preds, targets)
            fit += performance_metrics["fit"] * batch_size

            total += batch_size

    strategy.exit_eval(model) if strategy else None

    avg_loss = total_loss / max(total, 1)
    avg_fit = fit / max(total, 1)

    return avg_loss, avg_fit


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
        log.info(f"Saved: {acc_file[0]}")
    else:
        log.error("Best accuracy model not found")
        sys.exit(1)

    if loss_file:
        best_loss = torch.load(f=loss_file[0], weights_only=False)
        log.info(f"Saved: {loss_file[0]}")
    else:
        log.error("Best loss model not found")
        sys.exit(1)

    model_acc.load_state_dict(best_acc["model_state_dict"])
    model_loss.load_state_dict(best_loss["model_state_dict"])

    strategy_acc = make_strategy(cfg)
    strategy_acc.setup_for_eval(model_acc)
    loss_function = strategy_acc.loss_fn()

    test_loss_a, test_fit_a = _evaluate(
        model_acc,
        test_loader,
        loss_function,
        device,
        strategy=strategy_acc,
        wandb_logger=wandb_logger,
    )

    strategy_loss = make_strategy(cfg)
    strategy_loss.setup_for_eval(model_loss)
    test_loss_l, test_fit_l = _evaluate(
        model_loss,
        test_loader,
        loss_function,
        device,
        strategy=strategy_loss,
        wandb_logger=wandb_logger,
    )

    log.info(
        f"Best Model (by Accuracy) - Loss: {test_loss_a:.5f}, FIT: {test_fit_a:.3f}"
    )

    log.info(f"Best Model (by Loss) - Loss: {test_loss_l:.5f}, FIT: {test_fit_l:.3f}")

    compute_ls_mmse(
        test_loader=test_loader,
        device=device,
        model_acc=model_acc,
        model_loss=model_loss,
        base_dir=base_dir,
    )

    if not cfg.save_model:
        acc_file[0].unlink(missing_ok=True)
        loss_file[0].unlink(missing_ok=True)
        log.info("Deleted best models.")


if __name__ == "__main__":
    main()
