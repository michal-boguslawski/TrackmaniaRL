"""Policy evaluation utilities: run inference and log results.

Provides run_inference() for deterministic evaluation episodes and
log_evaluation_results() for logging per-episode and aggregate metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from collections.abc import Mapping

import numpy as np
import torch as T
from gymnasium.vector import VectorEnv

from rl_lib.agent import Agent
from rl_lib.tracking.base import MetricsLogger


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


def _episode_stat_from_info(
    info: dict,
    env_index: int,
    stats_key: str,
    stat_name: str,
) -> float | None:
    """Read an episode statistic from Gymnasium episode info.

    Tries the locations where RecordEpisodeStatistics may store a stat, in
    decreasing specificity:

    1. ``final_info[env_index][stats_key][stat_name]`` (NEXT_STEP autoreset,
       which keeps terminal infos nested per env)
    2. ``final_info[stats_key][stat_name][env_index]`` gated by
       ``final_info[f"_{stats_key}"]`` (SAME_STEP autoreset, which aggregates
       the terminal infos across envs instead of nesting them)
    3. ``info[stats_key][stat_name][env_index]`` gated by ``info[f"_{stats_key}"]``

    Args:
        info: Info dict from vector env step.
        env_index: Environment index in vector.
        stats_key: Episode statistics key (default "episode").
        stat_name: Name of the statistic to read (for example, "r" or "l").

    Returns:
        Episode statistic if found, None otherwise. None means the caller
        should fall back to accumulating rewards and step counts itself.
    """
    final_infos = info.get("final_info")
    try:
        nested_info = final_infos[env_index]
    except (IndexError, KeyError, TypeError):
        nested_info = None

    # Nested stats are unmasked; aggregated final_info requires a mask.
    sources = (
        (nested_info.get(stats_key) if isinstance(nested_info, Mapping) else None, None, False),
        (final_infos, f"_{stats_key}", True),
        (info, f"_{stats_key}", False),
    )
    for source, mask_key, require_mask in sources:
        if mask_key is None:
            stats, index = source, 0
        else:
            if not isinstance(source, Mapping):
                continue
            mask = source.get(mask_key)
            if mask is None:
                if require_mask:
                    continue
            else:
                try:
                    flags = np.asarray(mask).reshape(-1)
                    if not 0 <= env_index < flags.size or not bool(flags[env_index]):
                        continue
                except (IndexError, TypeError, ValueError):
                    continue
            stats, index = source.get(stats_key), env_index

        if not isinstance(stats, Mapping):
            continue
        try:
            values = np.asarray(stats.get(stat_name)).reshape(-1)
            if 0 <= index < values.size:
                return float(values[index])
        except (TypeError, ValueError):
            continue
    return None


def run_inference(
    agent: Agent,
    env: VectorEnv,
    episodes: int,
    seed: int | None = None,
    temperature: float | None = None,
    episode_stats_key: str = "episode",
) -> list[EpisodeResult]:
    """Run a policy until exactly ``episodes`` complete across a vector env.

    The agent's recurrent/temporal history is isolated to this run and its
    original train/eval mode is restored even if stepping raises an error.
    Episode returns and lengths are read from episode statistics in ``info``
    when present; accumulated step rewards and counts are the fallback.

    Args:
        agent: Policy agent (copied to isolate temporal state).
        env: Vectorized evaluation environment.
        episodes: Number of episodes to run (distributed across envs).
        seed: Random seed for environment reset.
        temperature: Action sampling temperature (None = agent default).
        episode_stats_key: Key for episode statistics in info dict.

    Returns:
        List of EpisodeResult with return and length for each episode.

    Raises:
        ValueError: If episodes < 1 or env.num_envs < 1.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if env.num_envs < 1:
        raise ValueError("inference environment must contain at least one env")

    was_training = agent.network.training
    agent.eval()
    agent.reset()
    try:
        observation, _ = env.reset(seed=seed)
        done = T.zeros(env.num_envs, dtype=T.bool, device=agent.device)
        episode_returns = np.zeros(env.num_envs, dtype=np.float64)
        episode_lengths = np.zeros(env.num_envs, dtype=np.int64)
        results: list[EpisodeResult] = []

        while len(results) < episodes:
            (
                next_observation,
                _,
                _,
                _,
                _,
                reward,
                _,
                _,
                done,
                info,
            ) = agent.step_env(
                env,
                observation,
                done,
                temperature=temperature,
            )
            episode_returns += np.asarray(reward, dtype=np.float64).reshape(env.num_envs)
            episode_lengths += 1
            done_mask = done.detach().cpu().numpy().astype(np.bool_, copy=False)

            for env_index in np.flatnonzero(done_mask):
                if len(results) == episodes:
                    break
                info_return = _episode_stat_from_info(
                    info, int(env_index), episode_stats_key, "r"
                )
                info_length = _episode_stat_from_info(
                    info, int(env_index), episode_stats_key, "l"
                )
                result = EpisodeResult(
                    return_=(
                        info_return
                        if info_return is not None
                        else float(episode_returns[env_index])
                    ),
                    length=(
                        int(info_length)
                        if info_length is not None
                        else int(episode_lengths[env_index])
                    ),
                )
                results.append(result)
                logger.debug(
                    "Inference episode %d/%d: return=%.3f length=%d",
                    len(results),
                    episodes,
                    result.return_,
                    result.length,
                )
                episode_returns[env_index] = 0.0
                episode_lengths[env_index] = 0

            observation = next_observation

        return results
    finally:
        agent.reset()
        if was_training:
            agent.train()
        else:
            agent.eval()


def log_evaluation_results(
    results: list[EpisodeResult],
    metrics_loggers: list[MetricsLogger],
    step: int,
    scope: str = "",
) -> dict[str, float]:
    """Log every completed episode and an aggregate summary to each logger.

    Metrics are prefixed with "evaluation/<scope>/" for organization.

    Args:
        results: List of EpisodeResult from run_inference.
        metrics_loggers: List of MetricsLogger backends.
        step: Training step for logging context.
        scope: Optional scope suffix (e.g., "periodic", "final", "video").

    Returns:
        Summary dict with mean/std/min/max returns and lengths.

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("cannot log an empty evaluation")

    prefix = "/".join(part.strip("/") for part in ("evaluation", scope) if part.strip("/"))
    for result in results:
        episode_metrics = {
            f"{prefix}/episode_return": result.return_,
            f"{prefix}/episode_length": float(result.length),
        }
        for metrics_logger in metrics_loggers:
            metrics_logger.log_metrics(episode_metrics, step=step)

    returns = np.asarray([result.return_ for result in results], dtype=np.float64)
    lengths = np.asarray([result.length for result in results], dtype=np.float64)
    summary = {
        f"{prefix}/return_mean": float(returns.mean()),
        f"{prefix}/return_std": float(returns.std()),
        f"{prefix}/return_min": float(returns.min()),
        f"{prefix}/return_max": float(returns.max()),
        f"{prefix}/length_mean": float(lengths.mean()),
        f"{prefix}/episodes": float(len(results)),
    }
    for metrics_logger in metrics_loggers:
        metrics_logger.log_metrics(summary, step=step)
    logger.info(
        "%s evaluation at step %d: episodes=%d return_mean=%.3f return_std=%.3f length_mean=%.1f",
        scope or "Model",
        step,
        len(results),
        summary[f"{prefix}/return_mean"],
        summary[f"{prefix}/return_std"],
        summary[f"{prefix}/length_mean"],
    )
    return summary
