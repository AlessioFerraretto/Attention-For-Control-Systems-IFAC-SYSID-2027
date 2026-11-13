from training_strategies.models.Transformer import Transformer
from training_strategies.task.FuturePrediction import FuturePrediction
from training_strategies.task.ImpulseResponsePrediction import ImpulseResponsePrediction
from training_strategies.task.KernelParametersEstimationOutputAlpha import (
    KernelParametersEstimationOutputAlpha,
)
from training_strategies.task.KernelParametersEstimationOutputAlphaC import (
    KernelParametersEstimationOutputAlphaC,
)
from training_strategies.task.KernelParametersEstimationOutputC import (
    KernelParametersEstimationOutputC,
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
    "kernel-parameters-estimation-input-uy-output-alpha-c": KernelParametersEstimationOutputAlphaC,
    "kernel-parameters-estimation-input-uy-alpha-output-c": KernelParametersEstimationOutputC,
    "kernel-parameters-estimation-input-uy-c-output-alpha": KernelParametersEstimationOutputAlpha,
    "kernel-parameters-estimation-input-g-output-alpha-c": KernelParametersEstimationOutputAlphaC,
    "kernel-parameters-estimation-input-g-alpha-output-c": KernelParametersEstimationOutputC,
    "kernel-parameters-estimation-input-g-c-output-alpha": KernelParametersEstimationOutputAlpha,
}
MODELS = {"transformer": Transformer}


def make_strategy(cfg) -> TrainingStrategy:
    task = TASKS[cfg.dataset.name](cfg)
    model_comp = MODELS[cfg.model.model_name]()
    return TrainingStrategy(task=task, model_comp=model_comp)
