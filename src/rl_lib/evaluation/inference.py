"""Policy evaluation utilities: run inference and log results.

Provides run_inference() for deterministic evaluation episodes and
log_evaluation_results() for logging per-episode and aggregate metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from logging import getLogger
from statistics import mean
from typing import Any

import torch as T
from gymnasium.vector import VectorEnv
from tqdm import tqdm

from rl_lib.policy.policy import Policy

logger = getLogger(__name__)


@dataclass(frozen=True)
class EpisodeResult:
    """Single completed episode result.

    Attributes:
        reward: Episode reward.
        length: Episode length in steps.
    """

    reward: float
    length: int


def get_results_from_info(
    info: dict[str, Any], stats_key: str = "episode"
) -> list[EpisodeResult]:
    episode_info = info.get("final_info")
    if not isinstance(episode_info, Mapping) or stats_key not in episode_info:
        episode_info = info

    mask_key = f"_{stats_key}"
    if stats_key in episode_info and mask_key in episode_info:
        dones = episode_info[mask_key]
        rewards = episode_info[stats_key]["r"][dones]
        lengths = episode_info[stats_key]["l"][dones]
        return [EpisodeResult(float(r), int(l)) for r, l in zip(rewards, lengths)]


def prepare_stats_from_results(
    results: list[EpisodeResult], prefix: str
) -> dict[str, float]:
    if not results:
        return {}

    return {
        f"{prefix}/reward": mean(r.reward for r in results),
        f"{prefix}/length": mean(r.length for r in results),
    }


def run_inference(
    policy: Policy,
    env: VectorEnv,
    episodes: int,
    seed: int | None = None,
    device: T.device | str = "cpu",
) -> list[EpisodeResult]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if env.num_envs < 1:
        raise ValueError("inference environment must contain at least one env")

    was_training = policy.training
    policy.eval()
    policy.reset()

    try:
        cnt = 0
        observation, _ = env.reset(seed=seed)
        done = T.zeros(env.num_envs, dtype=T.bool, device=device)
        episode_rewards = T.zeros(env.num_envs, dtype=T.float32, device=device)
        episode_lengths = T.zeros(env.num_envs, dtype=T.int32, device=device)
        results: list[EpisodeResult] = []

        with tqdm(total=episodes, desc="Evaluation", unit="episode") as progress:
            while len(results) < episodes:
                action, _ = policy.act(observation, done, deterministic=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                episode_rewards += reward
                episode_lengths += 1
                done = terminated | truncated

                temp = []
                if done.any():
                    temp = get_results_from_info(info)
                    if not temp:
                        for env_index in T.nonzero(done):
                            result = EpisodeResult(
                                reward=float(episode_rewards[env_index]),
                                length=int(episode_lengths[env_index]),
                            )
                            temp.append(result)
                    results.extend(temp)
                    progress.update(len(temp))

                    for result in temp:
                        cnt += 1
                        logger.debug(
                            "Inference episode %d/%d: reward=%.3f length=%d",
                            cnt,
                            episodes,
                            result.reward,
                            result.length,
                        )
                    episode_rewards[done] = 0.0
                    episode_lengths[done] = 0

                observation = next_state

        return results
    finally:
        if was_training:
            policy.train()
        else:
            policy.eval()
