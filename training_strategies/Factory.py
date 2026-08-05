from training_strategies.models.Transformer import Transformer
from training_strategies.task.FuturePrediction import FuturePrediction
from training_strategies.TrainingStrategy import TrainingStrategy

TASKS = {
    "fixed-future-prediction-v1": FuturePrediction,
    "fixed-future-prediction-v2": FuturePrediction,
    "dynamic-future-prediction-v1": FuturePrediction,
    "dynamic-future-prediction-v2": FuturePrediction,
    "dynamic-future-prediction-v3": FuturePrediction,
    "impulse-response-prediction": FuturePrediction,
    "kernel-parameters-prediction": FuturePrediction,
}
MODELS = {"transformer": Transformer}


def make_strategy(cfg) -> TrainingStrategy:
    task = TASKS[cfg.dataset.name]()
    # ssm = SSMS[cfg.model.ssm_module]()
    model_comp = MODELS[cfg.model.model_name]()
    return TrainingStrategy(task=task, model_comp=model_comp)
