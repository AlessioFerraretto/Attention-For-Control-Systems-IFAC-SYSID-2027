from collections.abc import Callable
from typing import Protocol


class TaskComponent(Protocol):
    """Cosa allenare: loss primaria e metriche."""

    def loss_fn(self) -> Callable: ...
    def compute_metrics(self, preds, targets) -> dict: ...  # acc, F1, ecc.
