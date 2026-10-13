"""Optimizer settings and parameter-group learning rates."""

from typing import Literal

from pydantic import Field

from rl_lib.training.settings.base import StrictModel


class OptimizerSettings(StrictModel):
    """Optimizer choice, parameter-group rates, and Adam-family options.

    Attributes:
        name: Adam or AdamW implementation.
        backbone_lr: Fallback rate for CNN and sequence encoder parameter groups.
        cnn_lr: Optional CNN-specific rate.
        sequence_encoder_lr: Optional sequence-encoder-specific rate.
        head_lr: Actor and critic head rate.
        weight_decay: Decay for regularized parameter groups.
        no_decay_weight_decay: Decay for bias and normalization groups.
        eps: Adam numerical stability constant.
        beta1: First-moment coefficient.
        beta2: Second-moment coefficient.
        amsgrad: Whether to enable AMSGrad.
        fused: Whether to use fused CUDA kernels when available.
    """

    name: Literal["AdamW", "Adam"] = Field(
        default="AdamW", description="Optimizer implementation."
    )
    backbone_lr: float = Field(
        default=1e-4,
        gt=0,
        description="Fallback rate for CNN and sequence encoder groups.",
    )
    cnn_lr: float | None = Field(
        default=None,
        gt=0,
        description="Optional CNN rate; defaults to backbone_lr.",
    )
    sequence_encoder_lr: float | None = Field(
        default=None,
        gt=0,
        description="Optional sequence-encoder rate; defaults to backbone_lr.",
    )
    head_lr: float = Field(
        default=1e-4,
        gt=0,
        description="Actor and critic head learning rate.",
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0,
        description="Weight decay for decay parameter groups.",
    )
    no_decay_weight_decay: float = Field(
        default=0.0,
        ge=0,
        description="Weight decay for bias and normalization groups.",
    )
    eps: float = Field(default=1e-5, gt=0, description="Adam numerical stability epsilon.")
    beta1: float = Field(default=0.9, ge=0, lt=1, description="Adam first-moment coefficient.")
    beta2: float = Field(default=0.999, ge=0, lt=1, description="Adam second-moment coefficient.")
    amsgrad: bool = Field(default=False, description="Enable the Adam AMSGrad variant.")
    fused: bool = Field(
        default=True,
        description="Use fused Adam/AdamW kernels on CUDA; ignored on CPU.",
    )
