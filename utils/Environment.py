"""Utility functions related to the environment/device running the training.

This module provides helpers to detect available accelerators (CUDA, MPS)
and to query runtime environment variables such as Slurm job identifiers.
"""

import logging
import os
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_torch_device() -> torch.device:
    """Get the preferred Torch device based on available accelerators.

    Returns:
        torch.device: The selected torch device. Preference order: CUDA, MPS, CPU.

    Notes:
        The function logs the detected backend and caches the result using
        `functools.lru_cache` to avoid repeated detection overhead.
    """
    if torch.cuda.is_available():
        cuda_v = torch.version.cuda
        cudnn_v = torch.backends.cudnn.version()
        gpu_n = torch.cuda.device_count()
        logger.info(
            f"Device: Using CUDA ({cuda_v}) | cudnn ({cudnn_v}) | GPUs: {gpu_n}"
        )
        return torch.device("cuda")

    # TODO: Can it even run on MPS or CPU or should we raise error?
    elif torch.backends.mps.is_available():
        logger.info("Device: Using MPS")
        return torch.device("mps")
    else:
        logger.info("Device: Using CPU")
        return torch.device("cpu")


def get_job_id() -> str:
    """Retrieve the Slurm job ID when running inside a Slurm allocation.

    Returns:
        str: The Slurm job id from the `SLURM_JOB_ID` environment variable, or
            the string `'DATA'` when running locally with no Slurm job id.

    Notes:
        The function logs whether execution is local or under Slurm.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        logger.info("Training on local machine.")
        return "DATA"

    logger.info(f"Training on Slurm cluster. Job ID: {job_id}")
    return job_id
