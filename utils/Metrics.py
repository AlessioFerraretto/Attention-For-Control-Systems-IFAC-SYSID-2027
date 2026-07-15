"""Helpers to extract monitoring metrics from `TrackableModule` instances.

The project uses `TrackableModule` for instrumented modules that expose
runtime metrics. This module provides a convenience function to gather
those metrics alongside parameter and gradient norms.
"""

from monitor.TrackableModule import TrackableModule


def collect_metrics(model):
    """Collect metrics from all `TrackableModule` instances within a model.

    Parameters:
        model : torch.nn.Module
            The top-level model to inspect for `TrackableModule` submodules.

    Returns:
        dict: Mapping of metric names to scalar values. Metric keys use the
            pattern `module_path.metric_name` for metrics provided by modules
            and `module_path.param_norm` / `module_path.param_grad_norm` for
            parameter norms.

    Notes:
        Only top-level parameters (recurse=False) for each `TrackableModule` are
        inspected for norms. If a parameter has no gradient, a gradient norm
        entry is not added for that parameter.
    """
    metrics = {}

    for name, module in model.named_modules():
        if isinstance(module, TrackableModule):
            data = module.get_metrics()
            if data:
                for k, v in data.items():
                    metrics[f"{name}.{k}"] = v

            # parameter and gradient norms for top-level parameters
            for pname, param in module.named_parameters(recurse=False):
                metrics[f"{name}.{pname}_norm"] = param.data.norm(2).item()
                if param.grad is not None:
                    metrics[f"{name}.{pname}_grad_norm"] = param.grad.data.norm(
                        2
                    ).item()

    return metrics
