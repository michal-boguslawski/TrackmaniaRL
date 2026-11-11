from pydantic import Field, PositiveInt

from rl_lib.config_base import StrictModel


class RolloutSettings(StrictModel):
    buffer_size: PositiveInt = 1024
    epochs: PositiveInt = 3
    minibatch_size: PositiveInt = 512
    gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
