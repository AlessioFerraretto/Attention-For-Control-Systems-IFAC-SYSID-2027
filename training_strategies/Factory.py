from training_strategies.models.Transformer import Transformer
from training_strategies.task.FuturePrediction import FuturePrediction
from training_strategies.task.ImpulseResponsePrediction import ImpulseResponsePrediction
from training_strategies.task.KernelParametersEstimation import (
    KernelParametersEstimation,
)
from training_strategies.TrainingStrategy import TrainingStrategy

TASKS = {
    "fixed-future-prediction-v1": FuturePrediction,
    "fixed-future-prediction-v2": FuturePrediction,
    "dynamic-future-prediction-v1": FuturePrediction,
    "dynamic-future-prediction-v2": FuturePrediction,
    "dynamic-future-prediction-v3": FuturePrediction,
    "impulse-response-prediction": ImpulseResponsePrediction,
    "dynamic-impulse-response-prediction": ImpulseResponsePrediction,
    "kernel-parameters-estimation": KernelParametersEstimation,
    "kernel-parameters-estimation-v2": KernelParametersEstimation,
    "kernel-parameters-estimation-v3": KernelParametersEstimation,
    "kernel-parameters-estimation-v4": KernelParametersEstimation,
    "kernel-parameters-estimation-v5": KernelParametersEstimation,
}
MODELS = {"transformer": Transformer}


def make_strategy(cfg) -> TrainingStrategy:
    task = TASKS[cfg.dataset.name](cfg)
    model_comp = MODELS[cfg.model.model_name]()
    return TrainingStrategy(task=task, model_comp=model_comp)
