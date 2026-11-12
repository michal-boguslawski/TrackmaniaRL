from pydantic import (
    PositiveFloat,
    PositiveInt,
)

from rl_lib.config_base import StrictModel


class TemporalConfig(StrictModel):
    """Temporal encoder (TemporalCNN1D) configuration.

    Attributes:
        out_dim: Output feature dimension.
        normalize_input: Apply LayerNorm to input sequence.
        normalize_output: Apply LayerNorm to output.
        bias: Include bias in Conv1d.
        norm_eps: LayerNorm epsilon.
        norm_affine: LayerNorm learnable affine parameters.
    """

    out_dim: PositiveInt = 256
    normalize_input: bool = True
    normalize_output: bool = True
    bias: bool = True
    norm_eps: PositiveFloat = 1e-5
    norm_affine: bool = True
