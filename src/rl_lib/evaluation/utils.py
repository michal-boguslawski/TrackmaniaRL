from __future__ import annotations

from collections.abc import Mapping
from statistics import mean, stdev
from typing import Any

from rl_lib.evaluation.models import EpisodeResult


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


def prepare_mean_stats_from_results(
    results: list[EpisodeResult],
    prefix: str,
) -> dict[str, float]:
    if not results:
        return {}

    return {
        f"{prefix}/reward": mean(r.reward for r in results),
        f"{prefix}/length": mean(r.length for r in results),
    }


def prepare_results_summary(
    results: list[EpisodeResult],
    prefix: str,
) -> dict[str, float | dict[str, float] | list[dict[str, float]]]:
    if not results:
        return {}

    summary = {}
    summary["episodes"] = [
        {"length": result.length, "reward": result.reward} for result in results
    ]
    summary["summary"] = {
        f"{prefix}/episodes": len(results),
        f"{prefix}/length_mean": mean(result.length for result in results),
        f"{prefix}/reward_mean": mean(result.reward for result in results),
        f"{prefix}/reward_std": stdev(result.reward for result in results),
        f"{prefix}/reward_max": max(result.reward for result in results),
        f"{prefix}/reward_min": min(result.reward for result in results),
    }
    return summary
