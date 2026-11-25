from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeResult:
    """Single completed episode result.

    Attributes:
        reward: Episode reward.
        length: Episode length in steps.
    """

    reward: float
    length: int
