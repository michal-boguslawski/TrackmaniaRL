from typing import Literal

from pydantic import Field, PositiveInt

from rl_lib.config_base import StrictModel


class ModelCheckpointCallbackSettings(StrictModel):
    name: Literal["model_checkpoint"] = Field(
        default="model_checkpoint",
        description="Callback registry name.",
    )
    interval: PositiveInt = Field(
        default=100_000,
    )
