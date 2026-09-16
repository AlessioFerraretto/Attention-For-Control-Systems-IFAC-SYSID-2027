"""Centralized, independent random streams for data and transformer code."""

import secrets

import numpy as np
import torch


class SeedManager:
    """Singleton owning the dataset RNG stream and transformer seed setup."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._dataset_torch_generator = torch.Generator()
            cls._instance._dataset_numpy_generator = np.random.default_rng()
        return cls._instance

    def seed_dataset(self, seed: int | None) -> None:
        """Seed only the private generators used to create synthetic data."""
        self.dataset_seed = seed
        if seed is None:
            self._dataset_torch_generator = torch.Generator()
            self._dataset_torch_generator.manual_seed(secrets.randbits(63))
            self._dataset_numpy_generator = np.random.default_rng()
            return

        self._dataset_torch_generator.manual_seed(seed)
        self._dataset_numpy_generator = np.random.default_rng(seed)

    def seed_transformer(self, seed: int | None) -> None:
        """Seed PyTorch immediately before transformer parameters are created."""
        self.transformer_seed = seed
        if seed is None:
            return

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def dataset_rand(self, *size, **kwargs) -> torch.Tensor:
        """Sample uniform values from the private dataset PyTorch generator."""
        return torch.rand(*size, generator=self._dataset_torch_generator, **kwargs)

    def dataset_randn(self, *size, **kwargs) -> torch.Tensor:
        """Sample normal values from the private dataset PyTorch generator."""
        return torch.randn(*size, generator=self._dataset_torch_generator, **kwargs)

    def dataset_numpy_randn(self, *size) -> np.ndarray:
        """Sample normal values from the private dataset NumPy generator."""
        return self._dataset_numpy_generator.standard_normal(size)

    def dataset_multivariate_normal(self, covariance: torch.Tensor) -> torch.Tensor:
        """Sample a zero-mean Gaussian using the private dataset generator."""
        chol = torch.linalg.cholesky(covariance)
        standard_normal = self.dataset_randn(
            covariance.shape[0], dtype=covariance.dtype, device=covariance.device
        )
        return chol @ standard_normal

    @staticmethod
    def make_data_loader_generator(seed: int) -> torch.Generator:
        """Create an independent generator for deterministic worker seeding."""
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def transformer_randn(self, *size, **kwargs) -> torch.Tensor:
        """Sample from PyTorch's transformer stream seeded by ``seed_transformer``."""
        return torch.randn(*size, **kwargs)


def seed_data_worker(worker_id: int) -> None:
    """Initialize a worker's private data stream from its DataLoader seed."""
    del worker_id
    SeedManager().seed_dataset(torch.initial_seed() % (2**32))
