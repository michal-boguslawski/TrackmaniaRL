from typing import Literal

from pydantic import Field

from rl_lib.config_base import StrictModel
from rl_lib.tracking.callbacks.base import CallbackEvent


class MetricsCallbackSettings(StrictModel):
    name: Literal["metrics_logger"] = Field(
        default="metrics_logger",
        description="Callback registry name.",
    )
    granularity: Literal[
        CallbackEvent.TRAIN_END, CallbackEvent.EPOCH_END, CallbackEvent.MINIBATCH_END
    ] = CallbackEvent.TRAIN_END
