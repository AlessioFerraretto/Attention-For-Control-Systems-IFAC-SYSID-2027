"""Factory utilities for constructing neural network models from configuration."""

import logging

import torch
from torch import nn

from models.embeddings.ContinuousSignalProjection import ContinuousSignalProjection
from models.embeddings.TransformerPositionalEncoding import (
    TransformerPositionalEncoding,
)
from models.NNComposition import NNComposition
from models.output_layers.MLPClassifier import MLPClassifier
from models.output_layers.MLPClassifierForFuturePredictionV1 import (
    MLPClassifierForFuturePredictionV1,
)
from models.output_layers.MLPClassifierForFuturePredictionV2 import (
    MLPClassifierForFuturePredictionV2,
)
from models.output_layers.MLPClassifierForFuturePredictionV3 import (
    MLPClassifierForFuturePredictionV3,
)
from models.output_layers.MLPClassifierForImpulseResponsePrediction import (
    MLPClassifierForImpulseResponsePrediction,
)
from models.output_layers.MLPClassifierForKernelParametersPrediction import (
    MLPClassifierForKernelParametersPrediction,
)
from models.output_layers.MLPClassifierForKernelParametersPredictionV2 import (
    MLPClassifierForKernelParametersPredictionV2,
)
from models.output_layers.MLPClassifierForKernelParametersPredictionV3 import (
    MLPClassifierForKernelParametersPredictionV3,
)
from models.output_layers.MLPClassifierForKernelParametersPredictionV4 import (
    MLPClassifierForKernelParametersPredictionV4,
)
from models.output_layers.MLPClassifierForKernelParametersPredictionV5 import (
    MLPClassifierForKernelParametersPredictionV5,
)
from models.output_layers.TokenWiseReadoutForImpulseResponsePrediction import (
    TokenWiseReadoutForImpulseResponse,
)
from models.TransformerEncoderLayer import TransformerEncoderLayer
from models.TransformerFinalLayer import TransformerFinalLayer

logger = logging.getLogger(__name__)


def construct_model(cfg, device: torch.device) -> nn.Module:
    """Return a complete neural network using the selected model configuration.

    Parameters:
        cfg : Any
            Configuration object describing model, dataset, and training settings.
        device : torch.device
            Device on which the model will be instantiated.

    Returns:
        nn.Module: Composed neural network instance.
    """
    logger.info("Constructing model")

    layers = [
        _build_layer(
            cfg=cfg, device=device, layer_index=idx, num_layers=cfg.model.num_layers
        )
        for idx in range(cfg.model.num_layers)
    ]

    return NNComposition(
        embedding=_get_embedding(cfg=cfg, device=device),
        layers=layers,
        output_layer=_get_output_module(cfg=cfg, device=device),
    )


def _get_embedding(cfg, device: torch.device) -> nn.Embedding:
    """Select the proper embedding layer based on the dataset or task.

    Parameters:
        cfg : Any
            Configuration object containing model and dataset settings.
        device : torch.device
            Device on which the embedding layer should operate.

    Returns:
        nn.Embedding: The selected embedding layer.
    """

    if cfg.model.embedding == "continuous-signal-projection":
        return ContinuousSignalProjection()
    elif cfg.model.embedding == "transformer-positional-encoding":
        return TransformerPositionalEncoding(cfg, device=device)
    else:
        raise NotImplementedError(
            f"Embedding '{cfg.model.embedding}' not implemented yet."
        )


def _build_layer(
    cfg, device: torch.device, layer_index: int, num_layers: int
) -> nn.Module:
    """Build a single instance of the model's inner layers.

    This is typically a Mamba block, but it can also be an isolated SSM module or
    a smoothing-problem-inspired layer.

    Parameters:
        cfg : Any
            Configuration object containing model settings.
        device : torch.device
            Device on which the layer should be created.
        input_dim : int | None
            Optional input dimension override for the current layer.

    Returns:
        nn.Module: A single layer instance matching the configured layer type.
    """

    if cfg.model.layer_name == "transformer-encoder-layer":
        if layer_index != num_layers - 1:
            return TransformerEncoderLayer(cfg=cfg, device=device)
        else:
            return TransformerFinalLayer(cfg=cfg, device=device)

    else:
        raise NotImplementedError(
            f"Layer '{cfg.model.layer_name}' not implemented yet."
        )


def _get_output_module(cfg, device: torch.device) -> nn.Module:
    """Select the proper output module based on the dataset or task.

    Parameters:
        cfg : Any
            Configuration object with output layer and dataset settings.
        device : torch.device
            Device on which the output module should operate.

    Returns:
        nn.Module: The configured output module.
    """

    if cfg.model.out_layer == "mlp-classifier":
        return MLPClassifier(
            cfg.model.input_dim_classifier,
            cfg.model.hidden_dim_classifier,
            cfg.model.output_dim_classifier,
            cfg.model.out_bias1_classifier,
            cfg.model.out_bias2_classifier,
            device=device,
        )
    elif cfg.model.out_layer == "mlp-classifier-for-future-prediction-v1":
        return MLPClassifierForFuturePredictionV1(
            cfg=cfg,
            device=device,
        )
    elif cfg.model.out_layer == "mlp-classifier-for-future-prediction-v2":
        return MLPClassifierForFuturePredictionV2(
            cfg=cfg,
            device=device,
        )
    elif cfg.model.out_layer == "mlp-classifier-for-future-prediction-v3":
        return MLPClassifierForFuturePredictionV3(
            cfg=cfg,
            device=device,
        )
    elif cfg.model.out_layer == "mlp-classifier-for-impulse-response-prediction":
        return MLPClassifierForImpulseResponsePrediction(
            cfg=cfg,
            device=device,
        )
    elif cfg.model.out_layer == "mlp-classifier-for-kernel-parameters-estimation":
        return MLPClassifierForKernelParametersPrediction(
            cfg=cfg,
            device=device,
        )
    elif cfg.model.out_layer == "mlp-classifier-for-kernel-parameters-estimation-v2":
        return MLPClassifierForKernelParametersPredictionV2(
            cfg=cfg,
            device=device,
        )
    elif cfg.model.out_layer == "mlp-classifier-for-kernel-parameters-estimation-v3":
        return MLPClassifierForKernelParametersPredictionV3(
            cfg=cfg,
            device=device,
        )
    elif cfg.model.out_layer == "mlp-classifier-for-kernel-parameters-estimation-v4":
        return MLPClassifierForKernelParametersPredictionV4(
            cfg=cfg,
            device=device,
        )
    elif cfg.model.out_layer == "mlp-classifier-for-kernel-parameters-estimation-v5":
        return MLPClassifierForKernelParametersPredictionV5(
            cfg=cfg,
            device=device,
        )
    elif cfg.model.out_layer == "token-wise-readout":
        return TokenWiseReadoutForImpulseResponse(
            cfg=cfg,
            device=device,
        )
    else:
        raise NotImplementedError(
            f"Output Module '{cfg.model.out_layer}' not implemented yet."
        )
