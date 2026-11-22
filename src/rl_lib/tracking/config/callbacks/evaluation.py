from typing import Literal

from pydantic import Field, PositiveInt

from rl_lib.config_base import StrictModel
from rl_lib.tracking.callbacks.base import CallbackEvent


class EvaluationCallbackSettings(StrictModel):
    name: Literal["evaluation"] = Field(
        default="evaluation",
        description="Callback registry name.",
    )

    interval: PositiveInt = Field(default=1, gt=0)
    episodes: PositiveInt = Field(default=1, gt=0)
    num_envs: PositiveInt = Field(default=1, gt=0)
    on_event: CallbackEvent = Field(default="env_step")
