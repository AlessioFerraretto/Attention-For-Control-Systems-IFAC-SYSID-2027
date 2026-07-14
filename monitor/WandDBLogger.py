"""Weights & Biases logging integration.

This module provides a logger class for integrating Weights & Biases (wandb)
with training pipelines, enabling experiment tracking and metric logging.
"""

import wandb
from omegaconf import OmegaConf


class WandDBLogger:
    """Logger for tracking experiments with Weights & Biases.

    This class manages the initialization, logging, and cleanup of Weights & Biases
    runs based on configuration settings.

    Attributes:
      enabled : bool
        Whether Weights & Biases logging is enabled.
      run : wandb.Run or None
        The active wandb run object, or None if logging is disabled.
    """

    def __init__(self, cfg):
        """Initialize the logger.

        Parameters:
            cfg : DictConfig
                Configuration object containing wandb settings (enable, entity, project,
                dataset, model_name).
        """
        self.enabled = cfg.wandb.enable
        if self.enabled:
            hydra_config = OmegaConf.to_container(cfg, resolve=True)
            self.run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                config=hydra_config,
                # tags=cfg.wandb.tags,
                name=f"Dataset:{cfg.wandb.dataset}-Model:{cfg.wandb.model_name}",
            )
        else:
            self.run = None

    def log(self, metrics):
        """Log metrics.

        Parameters:
            metrics : dict
                Dictionary of metric names and values to log.
        """
        if self.enabled:
            wandb.log(metrics)

    def finish(self):
        """Finish the run.

        Closes the current wandb run if logging is enabled.
        """
        if self.enabled:
            wandb.finish()
