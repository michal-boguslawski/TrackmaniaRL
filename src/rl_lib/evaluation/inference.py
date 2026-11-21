"""Policy evaluation utilities: run inference and log results.

Provides run_inference() for deterministic evaluation episodes and
log_evaluation_results() for logging per-episode and aggregate metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from statistics import mean

import torch as T
from gymnasium.vector import VectorEnv
from tqdm import tqdm

from rl_lib.policy.policy import Policy

logger = getLogger(__name__)


@dataclass(frozen=True)
class EpisodeResult:
    """Single completed episode result.

    Attributes:
        return_: Episode return (sum of rewards).
        length: Episode length in steps.
    """

    return_: float
    length: int


def prepare_stats_from_results(
    results: list[EpisodeResult], prefix: str
) -> dict[str, float]:
    if not results:
        return {}

    return {
        f"{prefix}/return": mean(r.return_ for r in results),
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
        observation, _ = env.reset(seed=seed)
        done = T.zeros(env.num_envs, dtype=T.bool, device=device)
        episode_returns = T.zeros(env.num_envs, dtype=T.float32, device=device)
        episode_lengths = T.zeros(env.num_envs, dtype=T.int32, device=device)
        results: list[EpisodeResult] = []

        with tqdm(total=episodes, desc="Evaluation", unit="episode") as progress:
            while len(results) < episodes:
                action, _ = policy.act(observation, done, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_returns += reward
                episode_lengths += 1
                done = terminated | truncated

                for env_index in T.nonzero(done):
                    result = EpisodeResult(
                        return_=float(episode_returns[env_index]),
                        length=int(episode_lengths[env_index]),
                    )
                    results.append(result)
                    progress.update()
                    logger.debug(
                        "Inference episode %d/%d: return=%.3f length=%d",
                        len(results),
                        episodes,
                        result.return_,
                        result.length,
                    )
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0

                observation = next_state

        return results
    finally:
        if was_training:
            policy.train()
        else:
            policy.eval()
