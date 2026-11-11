from typing import Literal

from pydantic import Field, PositiveInt

from rl_lib.config_base import StrictModel


class FrameStackWrapperSettings(StrictModel):
    """Settings for stacking consecutive observations.

    Attributes:
        stack_size: Number of consecutive observations to combine.
        frame_padding_type: Strategy used to pad the initial frame history.
    """

    name: Literal["frame_stack"] = Field(
        default="frame_stack", description="Select the frame-stack wrapper."
    )
    stack_size: PositiveInt = Field(
        description="Number of consecutive frames to stack."
    )
    frame_padding_type: Literal["reset", "zero"] = Field(
        default="reset", description="How to pad the initial observation history."
    )
