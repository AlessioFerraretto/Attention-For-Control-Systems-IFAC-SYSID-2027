"""Checkpoint manager utilities.

Provides a small persistence layer for saving, loading, and removing
training checkpoints and best-performing model snapshots. The manager
stores artifacts under a shared base directory and keeps the
underlying PyTorch state dicts together with the metrics used to
evaluate model quality.
"""

import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils import ModelCriteria
from utils.SeedManager import SeedManager

log = logging.getLogger("script")


class CheckpointManager:
    """Manage periodic and best-model checkpoints for a training run.

    Parameters:
      base_dir : Path
        Root directory where checkpoint files are stored.

    Attributes:
      base_dir : Path
        Root directory used to create and locate saved checkpoints.
      checkpoints_dir : Path
        Directory used for epoch-based checkpoint snapshots.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.checkpoints_dir = base_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        opt: Optimizer,
        loss: float,
        accuracy: float,
        global_step: int = 0,
    ):
        """Save a checkpoint for a specific epoch.

        Parameters:
          epoch : int
            Epoch number used in the checkpoint filename.
          model : nn.Module
            Model instance whose state dictionary is persisted.
          opt : Optimizer
            Optimizer instance whose state dictionary is persisted.
          loss : float
            Training or validation loss stored alongside the model state.
          accuracy : float
            Accuracy value stored alongside the model state.
          global_step : int
            Global optimization step associated with the checkpoint.

        Returns:
          None: This method writes the checkpoint to disk and returns nothing.
        """
        model_path = (
            self.checkpoints_dir / f"model_{epoch}_{loss:.5f}_{accuracy:.3f}.pt"
        )
        torch.save(
            obj={
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss,
                "accuracy": accuracy,
                "rng_states": SeedManager().get_rng_states(),
            },
            f=model_path,
        )

    def save_best(
        self,
        model: nn.Module,
        criteria: ModelCriteria,
        opt: Optimizer,
        loss: float,
        accuracy: float,
        epoch: int = 0,
        global_step: int = 0,
    ):
        """Save the current best model for the selected optimization criterion.

        Parameters:
          model : nn.Module
            Model to persist as the best-performing checkpoint.
          criteria : BestCriteria
            Criterion used to identify the best model, either "acc" or "loss".
          opt : Optimizer
            Optimizer state stored alongside the model.
          loss : float
            Current loss value associated with the saved model.
          accuracy : float
            Current accuracy value associated with the saved model.
          epoch : int
            Epoch index of the best model snapshot.
          global_step : int
            Global training step corresponding to the snapshot.

        Returns:
          None: This method removes any previous best-model artifact and writes a new one.
        """
        for path in self.base_dir.glob(f"best_model_{criteria.value}*.pt"):
            log.debug(f"Removing previous best: {path}")
            path.unlink()

        model_path = (
            self.base_dir / f"best_model_{criteria}_{loss:.3f}_{accuracy:.3f}.pt"
        )
        torch.save(
            obj={
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss,
                "accuracy": accuracy,
                "rng_states": SeedManager().get_rng_states(),
            },
            f=model_path,
        )

    def load_best(self, model: nn.Module, criteria: ModelCriteria) -> nn.Module:
        """Load the best checkpoint for a criterion into the provided model.

        Parameters:
          model : nn.Module
            Model instance that receives the checkpoint weights.
          criteria : ModelCriteria
            Criterion name used to find the best-model artifact.

        Returns:
          nn.Module: The same model instance after loading the best checkpoint state.

        Raises:
          FileNotFoundError: Raised when no best-model checkpoint exists for the requested criterion.
        """
        # USE criteria.value TO EXTRACT "acc" OR "loss" FROM THE ENUM
        criteria_str = criteria.value

        files = list(self.base_dir.glob(f"best_model_{criteria_str}*.pt"))
        if not files:
            log.error(f"Best {criteria_str} model not found in {self.base_dir}")
            raise FileNotFoundError(f"Best model for {criteria_str} not found.")

        log.info(f"Loaded: {files[0]}")
        checkpoint = torch.load(f=files[0], weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def load_checkpoint(self, model: nn.Module, checkpoint_path: Path) -> nn.Module:
        """Load a specific checkpoint from a given path into the provided model.

        Parameters:
          model : nn.Module
            Model instance that receives the checkpoint weights.
          checkpoint_path : Path
            Exact path to the checkpoint file to load.

        Returns:
          nn.Module: The same model instance after loading the checkpoint state.

        Raises:
          FileNotFoundError: Raised when the specified checkpoint file does not exist.
        """
        if not checkpoint_path.exists():
            log.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        log.info(f"Loaded: {checkpoint_path}")
        checkpoint = torch.load(f=checkpoint_path, weights_only=False)

        # Handle both custom wrapper dictionaries and raw state_dicts
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model

    def delete_best_models(self):
        """Delete any saved best-model checkpoints from the base directory.

        Returns:
          None: This method removes matching checkpoint files from disk.
        """
        for path in self.base_dir.glob("best_model_*.pt"):
            path.unlink(missing_ok=True)

    def resume_training(
        self,
        model: nn.Module,
        opt: Optimizer,
        resume_path: Path,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> tuple[nn.Module, Optimizer, int, float, float]:
        """Restore a training state from the given checkpoint.

        Load the model weights, optimizer state, metrics, and the saved random
        state so that subsequent data generation resumes deterministically.

        Parameters:
          model : nn.Module
            Model instance that receives the saved weights.
          opt : Optimizer
            Optimizer instance that receives the saved state.
          resume_path : Path
            Exact path to the checkpoint file to restore.

        Returns:
          tuple[nn.Module, Optimizer, int, float, float]: A tuple containing the restored model, optimizer, start epoch, last loss, and last accuracy.
        """
        log.info(f"Loading checkpoint from: {resume_path}")

        checkpoint = torch.load(resume_path, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])

        last_epoch = checkpoint.get("epoch", 0)
        start_epoch = last_epoch + 1

        last_loss = checkpoint.get("loss", float("inf"))
        last_accuracy = checkpoint.get("accuracy", float("-inf"))

        if "rng_states" in checkpoint:
            SeedManager().set_rng_states(checkpoint["rng_states"])
            log.info(
                "RNG states successfully restored. Data generation will be fully deterministic."
            )
        else:
            log.warning(
                "No RNG states found in checkpoint! Data generation will start fresh."
            )

        # TODO: Seed every epoch so that the dataloader is in sync without
        # having to iterate through all previous epochs.
        # This is a temporary solution to ensure deterministic behavior.

        if start_epoch > 1 and (train_loader or val_loader):
            epochs_to_skip = start_epoch - 1
            log.info(
                f"Fast-forwarding DataLoaders by {epochs_to_skip} epoch(s) to sync RNG states. Please wait..."
            )
            for _ in range(epochs_to_skip):
                if train_loader:
                    for _ in train_loader:
                        pass
                if val_loader:
                    for _ in val_loader:
                        pass
            log.info("DataLoaders fully synced!")

        return model, opt, start_epoch, last_loss, last_accuracy
