"""Utilities for collecting and persisting test-time results.

This module stores per-key tensors collected across batches and writes a
final concatenated snapshot to disk for later analysis.
"""

from collections import defaultdict
from pathlib import Path

import torch


class TestResultsManager:
    """Aggregate and save tensors produced during evaluation.

    Parameters:
      base_dir : Path
        Directory used to store the saved result file.

    Attributes:
      base_dir : Path
        Directory where the final results artifact is written.
      results : defaultdict[str, list]
        Mapping of result name to collected tensor values for each batch.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results = defaultdict(list)

    def update(self, **kwargs):
        """Append the current batch outputs to the running result collection.

        Parameters:
          **kwargs
            Named tensor values to store. Each keyword is used as the result key.

        Returns:
          None: The tensors are detached and moved to CPU before being appended.
        """
        for key, value in kwargs.items():
            # Only process and store the value if it actually exists
            if value is not None:
                # Ensure the list exists for this key before appending
                if key not in self.results:
                    self.results[key] = []
                self.results[key].append(value.detach().cpu())

    def save(self, filename="test_impulse_responses.pt"):
        """Concatenate all collected tensors and write them to disk.

        Parameters:
          filename : str
            File name used for the output artifact. Defaults to "test_impulse_responses.pt".

        Returns:
          None: The aggregated dictionary is saved to the configured base directory.
        """
        concatenated = {key: torch.cat(values) for key, values in self.results.items()}
        torch.save(concatenated, self.base_dir / filename)
