from typing import Literal

from pydantic import Field, PositiveInt

from rl_lib.config_base import StrictModel


class PPOOptimizationSettings(StrictModel):
    minibatch_size: PositiveInt
    epochs: PositiveInt
    minibatch_indexing_mode: Literal["iid", "sequential"] = Field(
        default="iid",
        description="Minibatch sample ordering strategy.",
    )
