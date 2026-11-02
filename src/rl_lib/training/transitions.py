from dataclasses import dataclass
from typing import Self

import torch as T


@dataclass(slots=True, frozen=True)
class Transition:
    obs: T.Tensor  # (N,H,W,C) uint8
    action: T.Tensor
    reward: T.Tensor
    terminated: T.Tensor
    truncated: T.Tensor
    next_obs: T.Tensor  # final_obs for done envs, else next_state
    extras: dict[str, T.Tensor]  # PPO: log_prob, value
    info: dict

    @classmethod
    def from_step(
        cls,
        state: T.Tensor,
        action: T.Tensor,
        extras: dict[str, T.Tensor],
        next_state: T.Tensor,
        reward: T.Tensor,
        terminated: T.Tensor,
        truncated: T.Tensor,
        info: dict,
    ) -> Self:
        return cls(
            obs=state,
            action=action,
            reward=reward.to(T.float32),
            terminated=terminated,
            truncated=truncated,
            next_obs=next_state,
            extras=extras,
            info=info,
        )
