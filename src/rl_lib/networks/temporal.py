import torch as T
from torch import nn

from rl_lib.networks.utils import init_layer


class TemporalCNN1D(nn.Module):
    def __init__(self, stack_size: int):
        super().__init__()
        self._encoder = nn.Conv1d(
            256, 256, kernel_size=stack_size
        )
        self._norm = nn.LayerNorm(256)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = x.permute(0, 2, 1)
        x = self._encoder(x)
        x = x.permute(0, 2, 1)
        x.squeeze_(1)
        x = self._norm(x)
        return x
