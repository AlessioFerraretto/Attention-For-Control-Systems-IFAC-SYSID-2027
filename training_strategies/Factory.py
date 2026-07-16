from training_strategies.models.Transformer import Transformer
from training_strategies.task.StableSpline import StableSpline
from training_strategies.TrainingStrategy import TrainingStrategy

TASKS = {"stable-spline-kernel": StableSpline}
# SSMS = {"transformer": Transformer}
MODELS = {"transformer": Transformer}


def make_strategy(cfg) -> TrainingStrategy:
    task = TASKS[cfg.dataset.name]()
    # ssm = SSMS[cfg.model.ssm_module]()
    model_comp = MODELS[cfg.model.layer_name]()
    return TrainingStrategy(task=task, model_comp=model_comp)
