from pydantic import Field

from rl_lib.config_base import StrictModel


class WrapperSettings(StrictModel):
    """Base class for wrapper settings."""

    name: str = Field(..., description="Name of the wrapper.")
