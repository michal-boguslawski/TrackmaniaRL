from dataclasses import dataclass

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
