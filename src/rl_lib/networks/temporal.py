"""Temporal encoder: 1D convolution over stacked feature sequences.

Reduces a (batch, stack_size, feature_dim) sequence to (batch, out_dim)
via a single full-length Conv1d kernel, with optional LayerNorm.
"""

import torch as T
from torch import nn

from rl_lib.networks.config import TemporalConfig


class TemporalCNN1D(nn.Module):
    """1D convolution over temporal feature sequences.

    Uses a kernel of size `stack_size` to collapse the sequence dimension
    in one step. Input normalization and output normalization are optional
    LayerNorm layers.

    Attributes:
        stack_size (int): Number of frames in the input sequence (kernel size).
        in_dim (int): Input feature dimension per frame.
        cfg (TemporalConfig): Configuration for output dim, bias, normalization.
    """

    def __init__(self, stack_size: int, in_dim: int, config: TemporalConfig):
        super().__init__()
        self.stack_size = stack_size
        self.in_dim = in_dim
        self.cfg = config

        self._encoder = nn.Conv1d(in_dim, self.cfg.out_dim, kernel_size=stack_size, bias=self.cfg.bias)
        self._norm = nn.LayerNorm(in_dim, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine) if self.cfg.normalize_input else nn.Identity()
        self._out_norm = nn.LayerNorm(self.cfg.out_dim, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine) if self.cfg.normalize_output else nn.Identity()

    @property
    def out_dim(self) -> int:
        """Output feature dimension after temporal encoding."""
        return self.cfg.out_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Encode sequence of features to fixed-size vector.

        Args:
            x: Tensor of shape (batch, stack_size, in_dim).

        Returns:
            Tensor of shape (batch, out_dim).
        """
        x = self._norm(x)
        x = x.permute(0, 2, 1)  # (batch, in_dim, stack_size) for Conv1d
        x = self._encoder(x)
        x = x.permute(0, 2, 1)  # (batch, 1, out_dim)
        x = x.squeeze(1)        # (batch, out_dim)
        x = self._out_norm(x)
        return x

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in super().parameters())
        cfg_fields = ", ".join(f"{k}={v}" for k, v in self.cfg.model_dump().items())
        return (
            f"{self.__class__.__name__}("
            f"stack_size={self.stack_size}, "
            f"in_dim={self.in_dim}, "
            f"{cfg_fields}, "
            f"params={n_params:,})"
        )

    def config(self) -> dict[str, int | str]:
        """Hyperparameters + param count, suitable for mlflow.log_params (with a prefix)."""
        return {
            "stack_size": self.stack_size,
            "in_dim": self.in_dim,
            **self.cfg.model_dump(),
            "n_params": sum(p.numel() for p in super().parameters()),
        }
