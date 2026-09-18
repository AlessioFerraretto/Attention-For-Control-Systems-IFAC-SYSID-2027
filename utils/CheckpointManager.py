"""Checkpoint manager utilities.

Provides a small persistence layer for saving, loading, and removing
training checkpoints and best-performing model snapshots. The manager
stores artifacts under a shared base directory and keeps the
underlying PyTorch state dicts together with the metrics used to
evaluate model quality.
"""

import logging
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer

log = logging.getLogger("script")
BestCriteria = Literal["acc", "loss"]


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
        self, epoch: int, model: nn.Module, opt: Optimizer, loss: float, accuracy: float
    ):
        """Save a standard checkpoint for a specific epoch.

        Parameters:
          epoch : int
            Epoch number used in the checkpoint filename.
          model : nn.Module
            Model instance whose state dictionary is persisted.
          opt : Optimizer
            Optimizer instance whose state dictionary is persisted.
          loss : float
            Training or validation loss to store alongside the model state.
          accuracy : float
            Accuracy value to store alongside the model state.

        Returns:
          None: This method writes the checkpoint to disk and returns nothing.
        """
        model_path = (
            self.checkpoints_dir / f"model_{epoch}_{loss:.5f}_{accuracy:.3f}.pt"
        )
        torch.save(
            obj={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss,
                "accuracy": accuracy,
            },
            f=model_path,
        )

    def save_best(
        self,
        model: nn.Module,
        criteria: BestCriteria,
        opt: Optimizer,
        loss: float,
        accuracy: float,
    ):
        """Save the current best model for the requested optimization criterion.

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

        Returns:
          None: This method removes any previous best-model artifact and writes a new one.
        """
        for path in self.base_dir.glob(f"best_model_{criteria}*.pt"):
            log.debug(f"Removing previous best: {path}")
            path.unlink()

        model_path = (
            self.base_dir / f"best_model_{criteria}_{loss:.3f}_{accuracy:.3f}.pt"
        )
        torch.save(
            obj={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            },
            f=model_path,
        )

    def load_best(self, model: nn.Module, criteria: BestCriteria) -> nn.Module:
        """Load the best checkpoint for a criterion into the provided model.

        Parameters:
          model : nn.Module
            Model instance that receives the checkpoint weights.
          criteria : BestCriteria
            Criterion name used to find the best-model artifact.

        Returns:
          nn.Module: The same model instance after loading the best checkpoint state.

        Raises:
          FileNotFoundError: Raised when no best-model checkpoint exists for the requested criterion.
        """
        files = list(self.base_dir.glob(f"best_model_{criteria}*.pt"))
        if not files:
            log.error(f"Best {criteria} model not found in {self.base_dir}")
            raise FileNotFoundError(f"Best model for {criteria} not found.")

        log.info(f"Loaded: {files[0]}")
        checkpoint = torch.load(f=files[0], weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def delete_best_models(self):
        """Delete any saved best-model checkpoints from the base directory.

        Returns:
          None: This method removes matching checkpoint files from disk.
        """
        for path in self.base_dir.glob("best_model_*.pt"):
            path.unlink(missing_ok=True)
