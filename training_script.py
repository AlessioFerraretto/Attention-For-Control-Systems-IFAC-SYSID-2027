"""
Modular training procedure that can manage custom training strategies
"""

import logging
import sys

# Windows compatibility: Python's default console encoding on Windows is often
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

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from datasets.Factory import get_data_loaders
from evaluate import evaluate_model
from models.Factory import construct_model
from monitor.WandDBLogger import WandDBLogger
from training_strategies.Factory import make_strategy
from utils.CheckpointManager import CheckpointManager
from utils.Environment import get_job_id, get_torch_device
from utils.Estimators import (
    compute_hutchinson_df_and_sure,
    compute_impulseest,
    compute_impulseest_df_and_sure,
    compute_ls,
    compute_ls_df_and_sure,
    compute_mmse,
    compute_mmse_df_and_sure,
    verify_sure_ranking,
)
from utils.ModelCriteria import ModelCriteria
from utils.SeedManager import SeedManager
from utils.TestResultsManager import TestResultsManager

log = logging.getLogger("script")

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

    model_mode = cfg.get("model_mode")
    running_mode = cfg.get("running_mode")

    match model_mode:
        case "new":
            model = construct_model(cfg=cfg, device=device)
        case "load":
            model_path_str = cfg.get("model_path")
            if not model_path_str:
                log.error(
                    "You must specify the model path. Example: model_path='outputs/model.pt'"
                )
                sys.exit(1)

            model_path = Path(model_path_str)
            if not model_path.exists():
                log.error(f"Model file not found: {model_path}")
                sys.exit(1)

            log.info(f"Loading weights from: {model_path}")
            model = construct_model(cfg=cfg, device=device)
            ckpt_manager = CheckpointManager(model_path.parent)
            model = ckpt_manager.load_checkpoint(model, model_path)

    match running_mode:
        case "training":
            training(
                model=model,
                cfg=cfg,
                base_dir=base_dir,
                device=device,
                wandb_logger=wandb_logger,
                train_loader=train_loader,
                val_loader=val_loader,
            )

            model_acc, model_loss = load_best_models(cfg, device, base_dir)

            evaluate_test_set(
                test_loader=test_loader,
                device=device,
                models={ModelCriteria.ACC: model_acc, ModelCriteria.LOSS: model_loss},
                cfg=cfg,
                base_dir=base_dir,
                wandb_logger=wandb_logger,
            )
        case "inference":
            evaluate_test_set(
                test_loader=test_loader,
                device=device,
                models={ModelCriteria.INF: model},
                cfg=cfg,
                base_dir=base_dir,
                wandb_logger=wandb_logger,
            )

        case "diagnostic":
            diagnostic(
                model=model,
                cfg=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
            )


def training(
    model: nn.Module,
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

    ckpt_manager = CheckpointManager(base_dir)

    # ------------------------------------STRATEGY AND MODEL----------------------------
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

    # Initialize start_epoch to 1 (will be overwritten if resuming)
    start_epoch = 1

    # Check if a resume checkpoint path is provided in the configuration
    resume_path = cfg.get("resume_checkpoint")
    if resume_path and Path(resume_path).exists():
        log.info(f"Resuming training from checkpoint: {resume_path}")
        model, opt, start_epoch, best_val_loss, best_fit = ckpt_manager.resume_training(
            model=model,
            opt=opt,
            resume_path=resume_path,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        # Synchronize the global step for schedulers or custom strategy logic
        global_step = (start_epoch - 1) * len(train_loader)
    elif resume_path:
        log.error(f"Resume path specified but not found: {resume_path}")
        sys.exit(1)

    # ----------------------------TRAINING, VALIDATION PROCEDURE-----------------------------
    for epoch in range(start_epoch, cfg.epochs + 1):
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
            ckpt_manager.save_checkpoint(epoch, model, opt, val_loss, fit)

        log.info("SAVE THINGS")

        # Save best model in terms of accuracy
        if fit > best_fit:
            log.info("SAVE THINGS 111111111")
            best_fit = fit
            ckpt_manager.save_best(model, ModelCriteria.ACC, opt, val_loss, fit)

        # Save best model in terms of loss
        if val_loss < best_val_loss:
            log.info("SAVE THINGS 222222222")
            best_val_loss = val_loss
            ckpt_manager.save_best(model, ModelCriteria.LOSS, opt, val_loss, fit)

    log.info("Training complete.")


def diagnostic(
    model: nn.Module,
    cfg: DictConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
):
    # 1. Setup strategy and loss function for evaluation
    strategy = make_strategy(cfg)
    strategy.setup_for_eval(model)
    loss_function = strategy.loss_fn()

    # 2. Evaluate on Training Set
    log.info("Evaluating on Training Set...")
    train_loss, train_fit = _evaluate(
        model=model,
        loader=train_loader,
        loss_fn=loss_function,
        device=device,
        strategy=strategy,
    )

    # 3. Evaluate on Validation Set
    log.info("Evaluating on Validation Set...")
    val_loss, val_fit = _evaluate(
        model=model,
        loader=val_loader,
        loss_fn=loss_function,
        device=device,
        strategy=strategy,
    )

    # 4. Evaluate on Test Set
    log.info("Evaluating on Test Set...")
    test_loss, test_fit = _evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_function,
        device=device,
        strategy=strategy,
    )

    # 5. Formatted output
    total_params = sum(p.numel() for p in model.parameters())
    arch_name = cfg.model.get("description", "Transformer")

    print("\n" + "=" * 90)
    print(" " * 30 + "DIAGNOSTIC FIT RESULTS")
    print("=" * 90)
    print(
        f"{'Architecture':<20} | {'# Parameters':<15} | {'Train FIT':<10} | {'Val FIT':<10} | {'Test FIT':<10}"
    )
    print("-" * 90)
    print(
        f"{arch_name:<20} | {total_params:<15,} | {train_fit:>9.3f}% | {val_fit:>9.3f}% | {test_fit:>9.3f}%"
    )
    print("=" * 90 + "\n")

    # 6. Quick diagnostic log
    if train_fit > 95.0 and test_fit < 85.0:
        log.info(
            "DIAGNOSIS: Hypothesis 2 (Benign Interpolation). The model interpolates the training set."
        )
    elif abs(train_fit - test_fit) < 10.0:
        log.info(
            "DIAGNOSIS: Hypothesis 1 or 3 (Population Bayes). The model does NOT interpolate the finite sample."
        )


def evaluate_test_set(
    test_loader: DataLoader,
    device: torch.device,
    models: dict[ModelCriteria, nn.Module],
    cfg: DictConfig,
    base_dir: Path,
    wandb_logger: WandDBLogger,
):
    """
    Evaluates all provided models (passed as a dict mapping criteria to model)
    and the LS/MMSE/Impulseest baselines in a single, unified loop over the test dataset.
    Also computes Stein's Unbiased Risk Estimate (SURE) and effective degrees of freedom.
    """
    log.info("Starting unified test set evaluation (Models + Baselines + SURE)...")
    results_manager = TestResultsManager(base_dir)

    # Set up the strategy using the first available model
    strategy = make_strategy(cfg)
    first_model = next(iter(models.values()))
    strategy.setup_for_eval(first_model)
    loss_fn = strategy.loss_fn()

    for model in models.values():
        model.eval()

    total_samples = 0

    # Group all estimators (models + baselines)
    estimators = list(models.keys()) + ["ls", "mmse", "impulseest"]
    totals = {
        "loss": {crit: 0.0 for crit in models},
        "fit": {k: 0.0 for k in estimators},
        "df": {k: 0.0 for k in estimators},
        "sure": {k: 0.0 for k in estimators},
    }

    with torch.no_grad():
        for inputs, targets, info, *rest in test_loader:
            batch_size = inputs.size(0)
            total_samples += batch_size

            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = rest[0].to(device) if rest else None

            u_batch = info["u"].to(device)
            y_batch = info["y"].to(device)
            noise_var_batch = info["noise_variance"].to(device)
            matrix_kernel_batch = info["kernel_matrix"].to(device)
            g_length = test_loader.dataset.g_length

            # --- MODEL PREDICTIONS & METRICS ---
            preds_dict = {}
            for crit, model in models.items():
                preds, _ = model(inputs, masks)
                preds_dict[crit] = preds

                loss = loss_fn(preds, targets).item() * batch_size
                totals["loss"][crit] += loss

                metrics = strategy.compute_metrics(preds, targets)
                totals["fit"][crit] += metrics["fit"] * batch_size

                # Compute DF and SURE (Temporarily re-enable gradients)
                with torch.enable_grad():
                    df_batch, sure_batch = compute_hutchinson_df_and_sure(
                        model,
                        inputs,
                        masks,
                        u_batch,
                        y_batch,
                        noise_var_batch,
                        g_length,
                        M=5,
                    )
                totals["df"][crit] += df_batch.sum().item()
                totals["sure"][crit] += sure_batch.sum().item()

            # --- BASELINES PREDICTIONS ---
            ls_estimate, fit_ls_batch, best_order = compute_ls(
                u_batch, y_batch, targets, g_length
            )
            mmse_estimate, fit_mmse_batch = compute_mmse(
                u_batch, y_batch, targets, noise_var_batch, matrix_kernel_batch
            )
            impulseest_estimate, fit_impulseest_batch = compute_impulseest(
                u_batch, y_batch, targets, noise_var_batch, matrix_kernel_batch
            )

            # --- BASELINES DF & SURE ---
            df_ls_batch, sure_ls_batch = compute_ls_df_and_sure(
                u_batch, y_batch, noise_var_batch, ls_estimate, best_order
            )
            df_mmse_batch, sure_mmse_batch = compute_mmse_df_and_sure(
                u_batch, y_batch, noise_var_batch, matrix_kernel_batch, g_length
            )
            df_impulseest_batch, sure_impulseest_batch = compute_impulseest_df_and_sure(
                u_batch,
                y_batch,
                noise_var_batch,
                matrix_kernel_batch,
                g_length,
            )

            # Accumulate Baseline Metrics
            totals["fit"]["ls"] += fit_ls_batch.sum().item()
            totals["df"]["ls"] += df_ls_batch.sum().item()
            totals["sure"]["ls"] += sure_ls_batch.sum().item()

            totals["fit"]["mmse"] += fit_mmse_batch.sum().item()
            totals["df"]["mmse"] += df_mmse_batch.sum().item()
            totals["sure"]["mmse"] += sure_mmse_batch.sum().item()

            totals["fit"]["impulseest"] += fit_impulseest_batch.sum().item()
            totals["df"]["impulseest"] += df_impulseest_batch.sum().item()
            totals["sure"]["impulseest"] += sure_impulseest_batch.sum().item()

            # --- STORE RESULTS ---
            results_manager.update(
                true_impulse_response=targets,
                ls_estimate=ls_estimate,
                mmse_estimate=mmse_estimate,
                impulseest_estimate=impulseest_estimate,
                model_best_accuracy_estimate=preds_dict.get(ModelCriteria.ACC),
                model_best_loss_estimate=preds_dict.get(ModelCriteria.LOSS),
                model_inference_estimate=preds_dict.get(ModelCriteria.INF),
                best_order=best_order,
            )

    # COMPUTE AVERAGES
    avgs = {
        metric_name: {k: val / max(total_samples, 1) for k, val in metric_dict.items()}
        for metric_name, metric_dict in totals.items()
    }

    # LOGGING
    print("\n" + "=" * 90)
    print(" " * 30 + "TEST SET EVALUATION SUMMARY")
    print("=" * 90)

    for crit in models:
        crit_name = crit.name if hasattr(crit, "name") else str(crit).upper()
        log.info(
            f"Model ({crit_name:<4}) - Loss: {avgs['loss'][crit]:.5f}, "
            f"FIT: {avgs['fit'][crit]:>6.3f}%, DF: {avgs['df'][crit]:>7.2f}, SURE: {avgs['sure'][crit]:>8.4f}"
        )

    for baseline, name in [
        ("impulseest", "Impulseest"),
        ("ls", "LS"),
        ("mmse", "MMSE"),
    ]:
        log.info(
            f"Baseline {name:<10} - FIT: {avgs['fit'][baseline]:>6.3f}%, "
            f"DF: {avgs['df'][baseline]:>7.2f}, SURE: {avgs['sure'][baseline]:>8.4f}"
        )
    print("=" * 90 + "\n")

    # WANDB LOGGING
    wandb_dict = {}

    # Log Baselines
    for baseline in ["ls", "mmse", "impulseest"]:
        wandb_dict[f"test/{baseline}_fit"] = avgs["fit"][baseline]
        wandb_dict[f"test/{baseline}_df"] = avgs["df"][baseline]
        wandb_dict[f"test/{baseline}_sure"] = avgs["sure"][baseline]

    # Log Models
    for crit in models:
        crit_val = crit.value if hasattr(crit, "value") else str(crit).lower()
        wandb_dict[f"test/model_{crit_val}_loss"] = avgs["loss"][crit]
        wandb_dict[f"test/model_{crit_val}_fit"] = avgs["fit"][crit]
        wandb_dict[f"test/model_{crit_val}_df"] = avgs["df"][crit]
        wandb_dict[f"test/model_{crit_val}_sure"] = avgs["sure"][crit]

    wandb_logger.log(wandb_dict)

    # RANKING VERIFICATION
    verify_sure_ranking(
        models=models,
        avg_sure=avgs["sure"],
        avg_fit=avgs["fit"],
        avg_fit_mmse=avgs["fit"]["mmse"],
        avg_sure_mmse=avgs["sure"]["mmse"],
        avg_fit_ls=avgs["fit"]["ls"],
        avg_sure_ls=avgs["sure"]["ls"],
        avg_fit_impulseest=avgs["fit"]["impulseest"],
        avg_sure_impulseest=avgs["sure"]["impulseest"],
    )

    results_manager.save()


def load_best_models(
    cfg: DictConfig, device: torch.device, base_dir: Path
) -> tuple[nn.Module, nn.Module]:
    """Build and load the two best models (ACC and LOSS) saved in the base_dir."""
    log.info("Loading best models (ACC and LOSS) for evaluation...")
    ckpt_manager = CheckpointManager(base_dir)

    model_acc = construct_model(cfg=cfg, device=device)
    model_acc = ckpt_manager.load_best(model_acc, ModelCriteria.ACC)

    model_loss = construct_model(cfg=cfg, device=device)
    model_loss = ckpt_manager.load_best(model_loss, ModelCriteria.LOSS)

    return model_acc, model_loss


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


if __name__ == "__main__":
    main()
