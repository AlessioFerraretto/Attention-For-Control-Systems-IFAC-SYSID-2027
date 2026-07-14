"""Trackable module for monitoring PyTorch model metrics.

This module provides a base class for PyTorch modules that support
metric tracking and monitoring capabilities.
"""

from torch import nn


class TrackableModule(nn.Module):
    """A PyTorch module with metric tracking capabilities.

    This is a base class that extends nn.Module to provide a standardized
    interface for modules that track and report metrics.
    """

    def get_metrics(self):
        """Get metrics from the module.

        Returns:
          dict: A dictionary containing metric names and values. Empty by default.
        """
        return {}
