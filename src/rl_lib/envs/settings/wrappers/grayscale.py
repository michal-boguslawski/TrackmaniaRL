from typing import Literal

from pydantic import Field

from rl_lib.config_base import StrictModel


class GrayscaleWrapperSettings(StrictModel):
    """Settings for converting observations to grayscale.

    Attributes:
        keep_dim: Whether to retain a one-channel dimension in the output.
    """

    name: Literal["grayscale"] = Field(
        default="grayscale", description="Select the grayscale observation wrapper."
    )
    keep_dim: bool = Field(
        default=True, description="Keep the grayscale channel dimension."
    )
