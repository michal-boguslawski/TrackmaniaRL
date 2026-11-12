from __future__ import annotations

from gymnasium import VectorizeMode
from gymnasium.vector import AutoresetMode
from pydantic import Field, PositiveInt

from rl_lib.config_base import StrictModel
from rl_lib.envs.settings.wrappers import WrapperSettings


class EnvironmentSettings(StrictModel):
    id: str = "CarRacing-v3"
    num_envs: PositiveInt = 8
    continuous: bool = True
    normalize_rewards: bool = True
    render_mode: str | None = None
    reward_normalization_gamma: float = Field(default=0.99, ge=0, le=1)
    reward_normalization_epsilon: float = Field(default=1e-8, gt=0)
    vectorization_mode: VectorizeMode = VectorizeMode.ASYNC
    autoreset_mode: AutoresetMode = AutoresetMode.SAME_STEP
    wrappers: list[WrapperSettings] = Field(default_factory=list)
