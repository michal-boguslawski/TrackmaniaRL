"""Benchmark rollout-buffer phases for one selected project worktree.

The benchmark driver is intentionally independent of the selected worktree's
source tree. Run this same file against a master worktree and an optimization
worktree to compare them without keeping a copy of the old implementation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import fields
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch as T


_TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from scripts.benchmarking import BenchmarkResult, benchmark_callables


def _positive_int(raw: str) -> int:
    """Parse a positive integer for CLI settings."""
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark configuration from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_TOOLS_ROOT,
        help="worktree whose rl_lib source should be benchmarked",
    )
    parser.add_argument("--label", default=None, help="label recorded in the report")
    parser.add_argument("--size", type=_positive_int, default=256, help="rollout steps (minimum 2)")
    parser.add_argument("--num-envs", type=_positive_int, default=16)
    parser.add_argument("--stack-size", type=_positive_int, default=4)
    parser.add_argument("--height", type=_positive_int, default=96)
    parser.add_argument("--width", type=_positive_int, default=96)
    parser.add_argument("--channels", type=_positive_int, default=1)
    parser.add_argument("--action-dim", type=_positive_int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=_positive_int, default=7)
    parser.add_argument("--threads", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--json", type=Path, help="optional path for JSON results")
    args = parser.parse_args(argv)
    if args.size < 2:
        parser.error("--size must be at least 2 to form a transition batch")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    return args


def _project_git_info(project_root: Path) -> dict[str, str | bool]:
    """Capture the checked-out revision and whether its worktree is dirty."""
    commit = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(project_root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "branch": branch or "detached", "dirty": bool(status)}


def _load_project_api(
    project_root: Path,
) -> tuple[type[Any], type[Any], type[Any] | None, set[str]]:
    """Import rollout APIs from the selected worktree, ahead of installed code."""
    source_root = project_root / "src"
    if not (source_root / "rl_lib").is_dir():
        raise ValueError(f"no src/rl_lib package found under {project_root}")
    sys.path.insert(0, str(source_root))

    # Import only after putting the selected worktree first on sys.path. This
    # allows this benchmark file to run unchanged against separate worktrees.
    from rl_lib.buffers.rollout_buffer import RolloutBuffer, RolloutStep
    try:
        from rl_lib.run_config import RolloutSettings
    except ImportError:
        RolloutSettings = None

    step_fields = {field.name for field in fields(RolloutStep)}
    return RolloutBuffer, RolloutStep, RolloutSettings, step_fields


def _make_steps(
    rollout_step_type: type[Any],
    step_fields: set[str],
    *,
    size: int,
    num_envs: int,
    observation_shape: tuple[int, int, int],
    action_dim: int,
    device: T.device,
    seed: int,
) -> list[Any]:
    """Build deterministic benchmark inputs on the requested device."""
    generator = T.Generator(device=device).manual_seed(seed)
    steps = []
    for _ in range(size):
        terminated = T.rand(num_envs, generator=generator, device=device) < 0.01
        truncated = T.rand(num_envs, generator=generator, device=device) < 0.02
        step_values = {
            "observation": T.randint(
                0,
                256,
                (num_envs, *observation_shape),
                dtype=T.uint8,
                generator=generator,
                device=device,
            ),
            "action": T.rand(
                (num_envs, action_dim), generator=generator, device=device
            ),
            "critic_value": T.rand(num_envs, generator=generator, device=device),
            "old_log_probs": T.rand(
                (num_envs, action_dim), generator=generator, device=device
            ),
            "reward": T.rand(num_envs, generator=generator, device=device),
            "terminated": terminated,
            "truncated": truncated,
        }
        if "truncated_value" in step_fields:
            step_values["truncated_value"] = T.rand(
                num_envs, generator=generator, device=device
            )
        steps.append(rollout_step_type(**step_values))
    return steps


def _fill_buffer(
    buffer_type: type[Any],
    settings_type: type[Any] | None,
    steps: list[Any],
    size: int,
    stack_size: int,
) -> Any:
    """Instantiate and fill the selected branch's rollout buffer."""
    if settings_type is None:
        buffer = buffer_type(size=size, stack_size=stack_size)
    else:
        buffer = buffer_type(settings_type(buffer_size=size), stack_size=stack_size)
    for step in steps:
        buffer.add(step)
    return buffer


def _check_batch(batch: dict[str, T.Tensor], args: argparse.Namespace) -> None:
    """Check expected PPO shapes and finite scalar training fields."""
    expected_shapes = {
        "observation": (
            args.num_envs,
            args.size + args.stack_size - 2,
            args.height,
            args.width,
            args.channels,
        ),
        "action": (args.num_envs, args.size - 1, args.action_dim),
        "old_log_probs": (args.num_envs, args.size - 1, args.action_dim),
        "critic_value": (args.num_envs, args.size - 1),
        "returns": (args.num_envs, args.size - 1),
        "advantages": (args.num_envs, args.size - 1),
        "dones": (args.num_envs, args.size + args.stack_size - 2),
    }
    for key, shape in expected_shapes.items():
        if tuple(batch[key].shape) != shape:
            raise AssertionError(f"{key} shape {tuple(batch[key].shape)} != {shape}")
    for key in ("critic_value", "returns", "advantages"):
        if not T.isfinite(batch[key]).all():
            raise AssertionError(f"{key} contains non-finite values")


def _resolve_device(requested: str) -> T.device:
    """Resolve the CLI device and reject unavailable CUDA devices."""
    if requested == "auto":
        return T.device("cuda" if T.cuda.is_available() else "cpu")
    device = T.device(requested)
    if device.type == "cuda" and not T.cuda.is_available():
        raise ValueError(f"requested CUDA device {device}, but CUDA is unavailable")
    return device


def _print_results(
    results: dict[str, BenchmarkResult],
) -> None:
    """Print phase summaries for one checked-out implementation."""
    print(f"{'phase':<16} {'median ms':>12} {'p90 ms':>12} {'peak CUDA MiB':>16}")
    for name, result in results.items():
        peak = (
            "n/a"
            if result.peak_cuda_bytes is None
            else f"{result.peak_cuda_bytes / (1024**2):.2f}"
        )
        print(f"{name:<16} {result.median_ms:>12.3f} {result.p90_ms:>12.3f} {peak:>16}")


def main(argv: list[str] | None = None) -> int:
    """Run rollout-buffer benchmarks for one branch/worktree."""
    args = _parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    if args.json is not None and not args.json.parent.is_dir():
        raise ValueError(f"JSON output directory does not exist: {args.json.parent}")

    git_info = _project_git_info(project_root)
    buffer_type, rollout_step_type, settings_type, step_fields = _load_project_api(
        project_root
    )
    device = _resolve_device(args.device)
    T.set_num_threads(args.threads)
    steps = _make_steps(
        rollout_step_type,
        step_fields,
        size=args.size,
        num_envs=args.num_envs,
        observation_shape=(args.height, args.width, args.channels),
        action_dim=args.action_dim,
        device=device,
        seed=args.seed,
    )
    ready_buffer = _fill_buffer(
        buffer_type, settings_type, steps, args.size, args.stack_size
    )
    _check_batch(ready_buffer.get(), args)

    operations: dict[str, Callable[[], object]] = {
        "add": lambda: _fill_buffer(
            buffer_type, settings_type, steps, args.size, args.stack_size
        ),
        "get": ready_buffer.get,
        "end_to_end": lambda: _fill_buffer(
            buffer_type, settings_type, steps, args.size, args.stack_size
        ).get(),
    }
    results = benchmark_callables(
        operations,
        warmups=args.warmups,
        repetitions=args.repetitions,
        device=device,
    )
    _print_results(results)

    label = args.label or git_info["branch"]
    config = {
        "label": label,
        "project_root": str(project_root),
        **git_info,
        "size": args.size,
        "num_envs": args.num_envs,
        "stack_size": args.stack_size,
        "observation_shape": [args.height, args.width, args.channels],
        "action_dim": args.action_dim,
        "device": str(device),
        "threads": T.get_num_threads(),
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "torch_version": T.__version__,
        "python_version": platform.python_version(),
    }
    print(
        f"branch={git_info['branch']} commit={git_info['commit']} "
        f"dirty={git_info['dirty']} label={label} device={device} "
        f"size={args.size} envs={args.num_envs} stack={args.stack_size}"
    )

    if args.json is not None:
        report = {
            "configuration": config,
            "results": {name: result.to_dict() for name, result in results.items()},
        }
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote benchmark report to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
