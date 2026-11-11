"""Shared base class for strict training settings models."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Configuration model that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid")
