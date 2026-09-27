"""Compare JSON reports emitted by the same benchmark on separate worktrees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_COMPARISON_CONFIG_KEYS = (
    "size",
    "num_envs",
    "stack_size",
    "observation_shape",
    "action_dim",
    "device",
    "threads",
    "warmups",
    "repetitions",
    "seed",
    "torch_version",
    "python_version",
)


def compare_report_data(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, str | float | int | None]]:
    """Compare matching phase results after checking benchmark configurations.

    Args:
        baseline: Decoded JSON report for the master/baseline worktree.
        candidate: Decoded JSON report for the optimized worktree.

    Returns:
        Per-phase latency speedup and CUDA-memory deltas.

    Raises:
        ValueError: If the reports use different workloads or result phases.
    """
    baseline_config = baseline["configuration"]
    candidate_config = candidate["configuration"]
    mismatched = [
        key
        for key in _COMPARISON_CONFIG_KEYS
        if baseline_config.get(key) != candidate_config.get(key)
    ]
    if mismatched:
        raise ValueError(
            "benchmark reports have mismatched configuration: "
            + ", ".join(mismatched)
        )

    baseline_results = baseline["results"]
    candidate_results = candidate["results"]
    if baseline_results.keys() != candidate_results.keys():
        raise ValueError("benchmark reports contain different phases")

    comparisons = []
    for phase in baseline_results:
        baseline_result = baseline_results[phase]
        candidate_result = candidate_results[phase]
        baseline_median = float(baseline_result["median_ms"])
        candidate_median = float(candidate_result["median_ms"])
        baseline_memory = baseline_result["peak_cuda_bytes"]
        candidate_memory = candidate_result["peak_cuda_bytes"]
        comparisons.append(
            {
                "phase": phase,
                "baseline_median_ms": baseline_median,
                "candidate_median_ms": candidate_median,
                "speedup": baseline_median / candidate_median,
                "baseline_peak_cuda_bytes": baseline_memory,
                "candidate_peak_cuda_bytes": candidate_memory,
                "peak_cuda_bytes_delta": (
                    None
                    if baseline_memory is None or candidate_memory is None
                    else candidate_memory - baseline_memory
                ),
            }
        )
    return comparisons


def main(argv: list[str] | None = None) -> int:
    """Load and print a pair of benchmark JSON reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    comparisons = compare_report_data(baseline, candidate)

    baseline_config = baseline["configuration"]
    candidate_config = candidate["configuration"]
    print(
        f"baseline={baseline_config['label']} ({baseline_config['commit'][:12]})  "
        f"candidate={candidate_config['label']} ({candidate_config['commit'][:12]})"
    )
    print(f"{'phase':<16} {'baseline ms':>13} {'candidate ms':>14} {'speedup':>10} {'CUDA MiB Δ':>12}")
    for comparison in comparisons:
        memory_delta = comparison["peak_cuda_bytes_delta"]
        memory_text = (
            "n/a"
            if memory_delta is None
            else f"{memory_delta / (1024**2):+.2f}"
        )
        print(
            f"{comparison['phase']:<16} "
            f"{comparison['baseline_median_ms']:>13.3f} "
            f"{comparison['candidate_median_ms']:>14.3f} "
            f"{comparison['speedup']:>9.2f}x "
            f"{memory_text:>12}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
