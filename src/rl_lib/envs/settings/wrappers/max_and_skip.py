from typing import Literal

from pydantic import Field, PositiveInt

from rl_lib.envs.settings.wrappers.base import BaseWrapperSettings


class MaxAndSkipWrapperSettings(BaseWrapperSettings):
    """Settings for repeating actions and max-pooling observations.

    Attributes:
        skip: Frames per action, defaulting to the environment's skip setting.
    """

    name: Literal["max_and_skip"] = Field(
        default="max_and_skip", description="Select the max-and-skip wrapper."
    )
    skip: PositiveInt | None = Field(
        default=None,
        description="Frames per action; defaults to EnvironmentSettings.skip.",
    )
