"""Dataset factory utility.

This module provides an helper for building training, validation, and
testing data loaders from a Hydra configuration object. It supports CIFAR-10,
MNIST, flattened MNIST, and sequence-to-sequence IH datasets with stratified
splits for the train/validation partitions.
"""

import logging

from torch.utils.data import DataLoader

from datasets.InductionHead import InductionHead

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

    if cfg.dataset.name == "induction-head":
        # TODO
        train_dataset = InductionHead(cfg=cfg)
        validation_dataset = InductionHead(cfg=cfg)
        test_dataset = InductionHead(cfg=cfg)
    else:
        raise NotImplementedError(
            f"Dataset '{cfg.dataset.name}' is not implemented yet."
        )

    # Initialize PyTorch DataLoaders for all splits
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training_batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.dataset.shuffle,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.validation_batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.dataset.shuffle,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.dataset.shuffle,
        pin_memory=True,
    )

    return train_loader, validation_loader, test_loader
