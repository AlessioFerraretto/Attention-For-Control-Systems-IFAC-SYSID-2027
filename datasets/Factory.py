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
from datasets.DynamicImpulseResponsePrediction import DynamicImpulseResponsePrediction
from datasets.FixedFuturePredictionV1 import FixedFuturePredictionV1
from datasets.FixedFuturePredictionV2 import FixedFuturePredictionV2
from datasets.ImpulseResponsePrediction import ImpulseResponsePrediction
from datasets.KernelParametersPredictionInputGAlphaOutputC import (
    KernelParametersPredictionInputGAlphaOutputC,
)
from datasets.KernelParametersPredictionInputGCOutputAlpha import (
    KernelParametersPredictionInputGCOutputAlpha,
)
from datasets.KernelParametersPredictionInputGOutputAlphaC import (
    KernelParametersPredictionInputGOutputAlphaC,
)
from datasets.KernelParametersPredictionInputUYAlphaOutputC import (
    KernelParametersPredictionInputUYAlphaOutputC,
)
from datasets.KernelParametersPredictionInputUYCOutputAlpha import (
    KernelParametersPredictionInputUYCOutputAlpha,
)
from datasets.KernelParametersPredictionInputUYOutputAlphaC import (
    KernelParametersPredictionInputUYOutputAlphaC,
)
from utils.SeedManager import SeedManager, seed_data_worker

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

    fixed_alpha = (
        cfg.dataset.alpha_max - cfg.dataset.alpha_min
    ) * SeedManager().dataset_rand(1).item() + cfg.dataset.alpha_min
    fixed_c = (cfg.dataset.c_max - cfg.dataset.c_min) * SeedManager().dataset_rand(
        1
    ).item() + cfg.dataset.c_min

    if cfg.data_seed is not None:
        loader_generators = (
            SeedManager().make_data_loader_generator(cfg.data_seed + offset)
            for offset in (0, 1, 2)
        )
        train_generator, validation_generator, test_generator = loader_generators

        loader_kwargs = {
            "worker_init_fn": seed_data_worker,
        }
    else:
        train_generator = validation_generator = test_generator = None
        loader_kwargs = {}

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
    elif cfg.dataset.name == "dynamic-impulse-response-prediction":
        train_dataset = DynamicImpulseResponsePrediction(
            cfg=cfg,
            batch_size=cfg.training_batch_size,
        )
        validation_dataset = DynamicImpulseResponsePrediction(
            cfg=cfg,
            batch_size=cfg.validation_batch_size,
        )
        test_dataset = DynamicImpulseResponsePrediction(
            cfg=cfg,
            batch_size=cfg.test_batch_size,
        )
    elif cfg.dataset.name == "impulse-response-prediction":
        train_dataset = ImpulseResponsePrediction(
            cfg=cfg, batch_size=cfg.training_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        validation_dataset = ImpulseResponsePrediction(
            cfg=cfg, batch_size=cfg.validation_batch_size, alpha=fixed_alpha, c=fixed_c
        )
        test_dataset = ImpulseResponsePrediction(
            cfg=cfg, batch_size=cfg.test_batch_size, alpha=fixed_alpha, c=fixed_c
        )
    elif cfg.dataset.name == "kernel-parameters-estimation-input-uy-output-alpha-c":
        train_dataset = KernelParametersPredictionInputUYOutputAlphaC(
            cfg=cfg,
            batch_size=cfg.training_batch_size,
        )
        validation_dataset = KernelParametersPredictionInputUYOutputAlphaC(
            cfg=cfg,
            batch_size=cfg.validation_batch_size,
        )
        test_dataset = KernelParametersPredictionInputUYOutputAlphaC(
            cfg=cfg,
            batch_size=cfg.test_batch_size,
        )
    elif cfg.dataset.name == "kernel-parameters-estimation-input-uy-alpha-output-c":
        train_dataset = KernelParametersPredictionInputUYAlphaOutputC(
            cfg=cfg,
            batch_size=cfg.training_batch_size,
        )
        validation_dataset = KernelParametersPredictionInputUYAlphaOutputC(
            cfg=cfg,
            batch_size=cfg.validation_batch_size,
        )
        test_dataset = KernelParametersPredictionInputUYAlphaOutputC(
            cfg=cfg,
            batch_size=cfg.test_batch_size,
        )
    elif cfg.dataset.name == "kernel-parameters-estimation-input-uy-c-output-alpha":
        train_dataset = KernelParametersPredictionInputUYCOutputAlpha(
            cfg=cfg,
            batch_size=cfg.training_batch_size,
        )
        validation_dataset = KernelParametersPredictionInputUYCOutputAlpha(
            cfg=cfg,
            batch_size=cfg.validation_batch_size,
        )
        test_dataset = KernelParametersPredictionInputUYCOutputAlpha(
            cfg=cfg,
            batch_size=cfg.test_batch_size,
        )
    elif cfg.dataset.name == "kernel-parameters-estimation-input-g-output-alpha-c":
        train_dataset = KernelParametersPredictionInputGOutputAlphaC(
            cfg=cfg,
            batch_size=cfg.training_batch_size,
        )
        validation_dataset = KernelParametersPredictionInputGOutputAlphaC(
            cfg=cfg,
            batch_size=cfg.validation_batch_size,
        )
        test_dataset = KernelParametersPredictionInputGOutputAlphaC(
            cfg=cfg,
            batch_size=cfg.test_batch_size,
        )
    elif cfg.dataset.name == "kernel-parameters-estimation-input-g-alpha-output-c":
        train_dataset = KernelParametersPredictionInputGAlphaOutputC(
            cfg=cfg,
            batch_size=cfg.training_batch_size,
        )
        validation_dataset = KernelParametersPredictionInputGAlphaOutputC(
            cfg=cfg,
            batch_size=cfg.validation_batch_size,
        )
        test_dataset = KernelParametersPredictionInputGAlphaOutputC(
            cfg=cfg,
            batch_size=cfg.test_batch_size,
        )
    elif cfg.dataset.name == "kernel-parameters-estimation-input-g-c-output-alpha":
        train_dataset = KernelParametersPredictionInputGCOutputAlpha(
            cfg=cfg,
            batch_size=cfg.training_batch_size,
        )
        validation_dataset = KernelParametersPredictionInputGCOutputAlpha(
            cfg=cfg,
            batch_size=cfg.validation_batch_size,
        )
        test_dataset = KernelParametersPredictionInputGCOutputAlpha(
            cfg=cfg,
            batch_size=cfg.test_batch_size,
        )
    else:
        raise NotImplementedError(
            f"Dataset '{cfg.dataset.name}' is not implemented yet."
        )

    if cfg.num_workers > 0:
        loader_kwargs["multiprocessing_context"] = torch.multiprocessing.get_context(
            "spawn"
        )

    # Initialize PyTorch DataLoaders for all splits
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training_batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.dataset.shuffle,
        pin_memory=False,
        generator=train_generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.validation_batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.dataset.shuffle,
        pin_memory=False,
        generator=validation_generator,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.dataset.shuffle,
        pin_memory=False,
        generator=test_generator,
        **loader_kwargs,
    )

    return train_loader, validation_loader, test_loader
