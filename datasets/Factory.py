"""Dataset factory utility.

This module provides an helper for building training, validation, and
testing data loaders from a Hydra configuration object. It supports CIFAR-10,
MNIST, flattened MNIST, and sequence-to-sequence IH datasets with stratified
splits for the train/validation partitions.
"""

import logging

import torch
from torch.utils.data import DataLoader

from datasets.DynamicFuturePredictionV1 import DynamicFuturePredictionV1
from datasets.DynamicFuturePredictionV2 import DynamicFuturePredictionV2
from datasets.DynamicFuturePredictionV3 import DynamicFuturePredictionV3
from datasets.FixedFuturePredictionV1 import FixedFuturePredictionV1
from datasets.FixedFuturePredictionV2 import FixedFuturePredictionV2

logger = logging.getLogger(__name__)


def get_data_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return training, validation, and testing data loaders based on configuration.

    This function loads the specified dataset defined in the configuration object
    and splits it into training, validation and testing subsets using a stratified
    approach via scikit-learn's `train_test_split`. The stratification ensures
    that all subsets maintain the same class distribution as the original dataset.

    Parameters:
      cfg : DictConfig
        Configuration object containing dataset and training parameters.

    Returns:
      tuple[DataLoader, DataLoader, DataLoader]: Three data loaders for training,
        validation, and testing.

    Raises:
      NotImplementedError: If the dataset name in `cfg.dataset.name` is not recognized.
    """

    logger.info(f"Retrieving '{cfg.dataset.name}'")

    fixed_alpha = (cfg.dataset.alpha_max - cfg.dataset.alpha_min) * torch.rand(
        1
    ).item() + cfg.dataset.alpha_min
    fixed_c = (cfg.dataset.c_max - cfg.dataset.c_min) * torch.rand(
        1
    ).item() + cfg.dataset.c_min

    if cfg.dataset.name == "fixed-future-prediction-v1":
        train_dataset = FixedFuturePredictionV1(
            cfg=cfg, batch_size=cfg.training_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        validation_dataset = FixedFuturePredictionV1(
            cfg=cfg, batch_size=cfg.validation_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        test_dataset = FixedFuturePredictionV1(
            cfg=cfg, batch_size=cfg.test_batch_size, alpha=fixed_alpha, c=fixed_c
        )
    elif cfg.dataset.name == "dynamic-future-prediction-v1":
        train_dataset = DynamicFuturePredictionV1(
            cfg=cfg, batch_size=cfg.training_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        validation_dataset = DynamicFuturePredictionV1(
            cfg=cfg, batch_size=cfg.validation_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        test_dataset = DynamicFuturePredictionV1(
            cfg=cfg, batch_size=cfg.test_batch_size, alpha=fixed_alpha, c=fixed_c
        )
    elif cfg.dataset.name == "fixed-future-prediction-v2":
        train_dataset = FixedFuturePredictionV2(
            cfg=cfg, batch_size=cfg.training_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        validation_dataset = FixedFuturePredictionV2(
            cfg=cfg, batch_size=cfg.validation_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        test_dataset = FixedFuturePredictionV2(
            cfg=cfg, batch_size=cfg.test_batch_size, alpha=fixed_alpha, c=fixed_c
        )
    elif cfg.dataset.name == "dynamic-future-prediction-v2":
        train_dataset = DynamicFuturePredictionV2(
            cfg=cfg, batch_size=cfg.training_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        validation_dataset = DynamicFuturePredictionV2(
            cfg=cfg, batch_size=cfg.validation_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        test_dataset = DynamicFuturePredictionV2(
            cfg=cfg, batch_size=cfg.test_batch_size, alpha=fixed_alpha, c=fixed_c
        )
    elif cfg.dataset.name == "dynamic-future-prediction-v3":
        train_dataset = DynamicFuturePredictionV3(
            cfg=cfg, batch_size=cfg.training_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        validation_dataset = DynamicFuturePredictionV3(
            cfg=cfg, batch_size=cfg.validation_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        test_dataset = DynamicFuturePredictionV3(
            cfg=cfg, batch_size=cfg.test_batch_size, alpha=fixed_alpha, c=fixed_c
        )
    else:
        raise NotImplementedError(
            f"Dataset '{cfg.dataset.name}' is not implemented yet."
        )

    mp_context = torch.multiprocessing.get_context("spawn")

    # Initialize PyTorch DataLoaders for all splits
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training_batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.dataset.shuffle,
        pin_memory=False,
        multiprocessing_context=mp_context,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.validation_batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.dataset.shuffle,
        pin_memory=False,
        multiprocessing_context=mp_context,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.dataset.shuffle,
        pin_memory=False,
        multiprocessing_context=mp_context,
    )

    return train_loader, validation_loader, test_loader
