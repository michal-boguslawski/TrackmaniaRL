from pydantic import Field, PositiveInt

from rl_lib.config_base import StrictModel


class RolloutSettings(StrictModel):
    steps_per_env: PositiveInt = 1024
    gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
