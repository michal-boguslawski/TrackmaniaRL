from typing import Literal

from pydantic import Field, FiniteFloat

from rl_lib.config_base import StrictModel


class RewardOnDoneWrapperSettings(StrictModel):
    """Reward overrides applied when an episode terminates or truncates.

    Attributes:
        on_terminated: Optional reward assigned on termination.
        on_truncated: Optional reward assigned on truncation.
        on_win: Optional reward assigned on a successful finish.
    """

    name: Literal["reward_on_done"] = Field(
        default="reward_on_done", description="Select the episode-end reward wrapper."
    )
    on_terminated: FiniteFloat | None = Field(
        default=None, description="Optional reward assigned on termination."
    )
    on_truncated: FiniteFloat | None = Field(
        default=-10.0, description="Optional reward assigned on truncation."
    )
    on_win: FiniteFloat | None = Field(
        default=100.0, description="Optional reward assigned on a successful finish."
    )
