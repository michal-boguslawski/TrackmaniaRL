"""Algorithm-specific loss settings."""

from pydantic import Field

from rl_lib.config_base import StrictModel


class EntropySettings(StrictModel):
    """Initial entropy bonus and its per-update decay schedule."""

    coefficient: float = Field(
        default=0.001, ge=0, description="Initial entropy bonus coefficient."
    )
    decay: float = Field(
        default=0.995,
        gt=0,
        le=1,
        description="Multiplicative entropy coefficient decay per update.",
    )
    min_coefficient: float = Field(
        default=1e-4,
        ge=0,
        description="Lower bound for the decayed entropy coefficient.",
    )
