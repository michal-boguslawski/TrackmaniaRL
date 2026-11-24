from typing import Literal

from pydantic import Field

from rl_lib.envs.settings.wrappers.base import BaseWrapperSettings


class CoupledGasBrakeWrapperSettings(BaseWrapperSettings):
    name: Literal["action_couple"] = Field(
        default="action_couple", description="Select the action_couple wrapper."
    )
