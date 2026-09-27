"""Measure what each metric verbosity level costs in a real PPO minibatch.

Wall-clock differences between verbosity levels are usually dominated by the
network forward/backward, so on CPU they sit inside the measurement noise. This
driver therefore records two things per level:

* timed phases, for the end-to-end minibatch cost (``train_step_v*``) and for
  the rollout-metric block on its own (``rollout_metrics_v*``), which is the
  only metric work that can be timed without the network hiding it; and
* exact counters, obtained by instrumenting the scalar-transfer helper: host
  transfers, scalars moved, metrics handed to the update loop, and metrics that
  survive the verbosity filter on the way to a tracking backend.

The counters are the actionable numbers; the timings confirm they cost what the
counters imply. Run it on a CUDA device to see the host transfers and skipped
kernels turn into real wall-clock gains.

Example::

    .venv/bin/python scripts/benchmarks/metric_verbosity.py --repetitions 5
    .venv/bin/python scripts/benchmarks/metric_verbosity.py \
        --profile tiny --transitions 512 --json logs/bench_verbosity.json
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch as T
import yaml


_TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from scripts.benchmarking import BenchmarkResult, benchmark_callables

import rl_lib
from rl_lib.agent import Agent
from rl_lib.networks.config import (
    ActorConfig,
    CNNConfig,
    ConvLayerConfig,
    CriticConfig,
    LinearLayerConfig,
    NetworkConfig,
    TemporalConfig,
)
from rl_lib.networks.factory import Network
from rl_lib.run_config import AgentSettings, TrainerSettings
from rl_lib.tracking.verbosity import Verbosity, filter_metrics
from rl_lib.training.ppo import losses as losses_module
from rl_lib.training.ppo import trainer as trainer_module
from rl_lib.training.ppo.trainer import PPOTrainer


_SHIPPED_CONFIG = Path(rl_lib.__file__).resolve().parent / "config" / "ppo_carracing.yaml"
_TINY_FEATURE_DIM = 4


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
        "--profile",
        choices=("production", "tiny"),
        default="production",
        help="production reads the shipped network config; tiny is a fast CPU stand-in",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_SHIPPED_CONFIG,
        help="training config supplying the production network section",
    )
    parser.add_argument(
        "--transitions",
        type=_positive_int,
        default=64,
        help="transitions per minibatch; observations hold stack_size frames each",
    )
    parser.add_argument("--stack-size", type=_positive_int, default=4)
    parser.add_argument("--action-dim", type=_positive_int, default=3)
    parser.add_argument("--height", type=_positive_int, default=96)
    parser.add_argument("--width", type=_positive_int, default=96)
    parser.add_argument("--channels", type=_positive_int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=_positive_int, default=5)
    parser.add_argument("--threads", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--json", type=Path, help="optional path for JSON results")
    args = parser.parse_args(argv)
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    return args


def _project_git_info() -> dict[str, str | bool]:
    """Capture the revision of the source tree that was actually imported."""
    repo_root = Path(rl_lib.__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": False}
    return {"commit": commit, "dirty": bool(status)}


def _network_config(profile: str, config_path: Path) -> NetworkConfig:
    """Build the network topology for the requested profile."""
    if profile == "production":
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return NetworkConfig(**raw["network"])
    hidden = [LinearLayerConfig(out_dim=8)]
    return NetworkConfig(
        cnn=CNNConfig(
            conv_layers=[
                ConvLayerConfig(out_channels=2, kernel_size=8, stride=4),
                ConvLayerConfig(out_channels=4, kernel_size=4, stride=2),
                ConvLayerConfig(out_channels=8, kernel_size=3),
                ConvLayerConfig(out_channels=8, kernel_size=3, stride=2),
            ],
            hidden_layers=hidden,
            out_dim=_TINY_FEATURE_DIM,
        ),
        temporal=TemporalConfig(out_dim=_TINY_FEATURE_DIM),
        actor=ActorConfig(hidden_layers=hidden),
        critic=CriticConfig(hidden_layers=hidden),
    )


def _make_agent(
    network_config: NetworkConfig, *, device: T.device, stack_size: int, action_dim: int
) -> Agent:
    """Build the policy/value agent every verbosity level shares.

    The levels are compared against identical weights, so they share one agent
    and each trainer only wraps it in its own optimizer.
    """
    network = Network(
        observation_dim=1,
        action_dim=action_dim,
        stack_size=stack_size,
        config=network_config,
    )
    return Agent(network=network, device=str(device), config=AgentSettings(stack_size=stack_size))


def _make_minibatch(
    *,
    transitions: int,
    stack_size: int,
    action_dim: int,
    observation_shape: tuple[int, int, int],
    device: T.device,
    seed: int,
) -> dict[str, T.Tensor]:
    """Build one minibatch shaped like ``get_iid_minibatches`` yields.

    Observation and done windows are flattened to ``transitions * stack_size``
    rows because that is what the rollout buffer hands to ``train_step``.
    """
    generator = T.Generator(device=device).manual_seed(seed)
    frames = transitions * stack_size
    return {
        "observation": T.randint(
            0, 256, (frames, *observation_shape), dtype=T.uint8, generator=generator, device=device
        ),
        "action": T.rand(
            (transitions, action_dim), generator=generator, device=device
        ),
        "old_log_probs": T.rand(
            (transitions, action_dim), generator=generator, device=device
        ),
        "old_values": T.rand(transitions, generator=generator, device=device),
        "returns": T.rand(transitions, generator=generator, device=device),
        "advantages": T.rand(transitions, generator=generator, device=device),
        "dones": T.zeros(frames, dtype=T.bool, device=device),
    }


def _make_rollout_batch(
    *,
    num_envs: int,
    size: int,
    action_dim: int,
    device: T.device,
    seed: int,
) -> dict[str, T.Tensor]:
    """Build one collected batch shaped like ``RolloutBuffer.get()``."""
    generator = T.Generator(device=device).manual_seed(seed)
    return {
        "action": T.rand((num_envs, size, action_dim), generator=generator, device=device),
        "old_log_probs": T.rand((num_envs, size, action_dim), generator=generator, device=device),
        "critic_value": T.rand((num_envs, size), generator=generator, device=device),
        "returns": T.rand((num_envs, size), generator=generator, device=device),
        "advantages": T.rand((num_envs, size), generator=generator, device=device),
    }


@contextmanager
def _counted_host_transfers() -> Iterator[dict[str, int]]:
    """Count scalar transfers and scalars moved inside one training step.

    ``_tensor_metrics_to_scalars`` is the single place where a device
    synchronization happens, so patching the module-level references is what
    turns "how much metric work did this level do" into a measurement instead of
    an estimate. The originals are restored on exit.
    """
    counts = {"host_transfers": 0, "scalars_moved": 0}
    original = losses_module._tensor_metrics_to_scalars

    def counting(metric_tensors: Mapping[str, T.Tensor]) -> dict[str, float]:
        counts["host_transfers"] += 1
        counts["scalars_moved"] += len(metric_tensors)
        return original(dict(metric_tensors))

    patched = (losses_module, trainer_module)
    saved = [(module, module._tensor_metrics_to_scalars) for module in patched]
    try:
        for module, _ in saved:
            module._tensor_metrics_to_scalars = counting  # type: ignore[attr-defined]
        yield counts
    finally:
        for module, function in saved:
            module._tensor_metrics_to_scalars = function  # type: ignore[attr-defined]


def _make_trainer(agent: Agent, verbosity: Verbosity) -> PPOTrainer:
    """Build one trainer for a verbosity level with its learning rate zeroed.

    Zeroing the rate keeps the weights fixed across the repeated measurements,
    so every repetition performs the same work instead of drifting further off
    the collected batch. The optimizer's per-step cost does not depend on the
    value of the rate.
    """
    trainer = PPOTrainer(agent, TrainerSettings(), verbosity=verbosity)
    for group in trainer._optimizer.param_groups:
        group["lr"] = 0.0
    return trainer


def _measure_metric_work(
    trainer: PPOTrainer, minibatch: Mapping[str, T.Tensor], verbosity: Verbosity
) -> dict[str, Any]:
    """Run one un-timed training step and count the metric work it performed."""
    with _counted_host_transfers() as counts:
        metrics = trainer.train_step(**minibatch)
    return {
        **counts,
        "metrics_returned": len(metrics),
        "metrics_to_backends": len(filter_metrics(metrics, verbosity)),
        "metric_keys": sorted(metrics),
    }


def _resolve_device(requested: str) -> T.device:
    """Resolve the CLI device and reject unavailable CUDA devices."""
    if requested == "auto":
        return T.device("cuda" if T.cuda.is_available() else "cpu")
    device = T.device(requested)
    if device.type == "cuda" and not T.cuda.is_available():
        raise ValueError(f"requested CUDA device {device}, but CUDA is unavailable")
    return device


def _print_timings(results: Mapping[str, BenchmarkResult]) -> None:
    """Print the paired latency summaries for every level and phase."""
    print(f"{'phase':<20} {'median ms':>12} {'p90 ms':>12} {'peak CUDA MiB':>16}")
    for name, result in results.items():
        peak = (
            "n/a"
            if result.peak_cuda_bytes is None
            else f"{result.peak_cuda_bytes / (1024**2):.2f}"
        )
        print(f"{name:<20} {result.median_ms:>12.3f} {result.p90_ms:>12.3f} {peak:>16}")


def _print_metric_work(metric_work: Mapping[str, Mapping[str, Any]]) -> None:
    """Print the exact per-minibatch metric counters for every level."""
    print(
        f"\n{'level':<6} {'host transfers':>16} {'scalars moved':>15} "
        f"{'metrics':>8} {'to backends':>12} {'rollout metrics':>16}"
    )
    for name, work in metric_work.items():
        print(
            f"{name:<6} {work['host_transfers']:>16} {work['scalars_moved']:>15} "
            f"{work['metrics_returned']:>8} {work['metrics_to_backends']:>12} "
            f"{work['rollout_metrics']:>16}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the verbosity benchmark across all three levels."""
    args = _parse_args(argv)
    if args.json is not None and not args.json.parent.is_dir():
        raise ValueError(f"JSON output directory does not exist: {args.json.parent}")

    device = _resolve_device(args.device)
    T.set_num_threads(args.threads)
    observation_shape = (args.height, args.width, args.channels)
    agent = _make_agent(
        _network_config(args.profile, args.config),
        device=device,
        stack_size=args.stack_size,
        action_dim=args.action_dim,
    )
    minibatch = _make_minibatch(
        transitions=args.transitions,
        stack_size=args.stack_size,
        action_dim=args.action_dim,
        observation_shape=observation_shape,
        device=device,
        seed=args.seed,
    )
    rollout_batch = _make_rollout_batch(
        num_envs=args.transitions,
        size=args.stack_size,
        action_dim=args.action_dim,
        device=device,
        seed=args.seed,
    )

    operations: dict[str, Callable[[], object]] = {}
    metric_work: dict[str, dict[str, Any]] = {}
    for verbosity in Verbosity:
        level = verbosity.value
        trainer = _make_trainer(agent, verbosity)
        operations[f"train_step_v{level}"] = lambda trainer=trainer: trainer.train_step(**minibatch)
        # Reaches into a private method on purpose: this is the only metric-only
        # code path, so it is what makes the difference visible on CPU.
        operations[f"rollout_metrics_v{level}"] = (
            lambda trainer=trainer: trainer._get_metrics_from_batch(rollout_batch)
        )
        metric_work[str(level)] = {
            **_measure_metric_work(trainer, minibatch, verbosity),
            "rollout_metrics": len(trainer._get_metrics_from_batch(rollout_batch)),
        }

    results = benchmark_callables(
        operations,
        warmups=args.warmups,
        repetitions=args.repetitions,
        device=device,
    )

    print(f"profile={args.profile} device={device} transitions={args.transitions} "
          f"frames={args.transitions * args.stack_size} threads={T.get_num_threads()}")
    _print_timings(results)
    _print_metric_work(metric_work)

    if args.json is not None:
        report = {
            "configuration": {
                **_project_git_info(),
                "label": args.profile,
                "profile": args.profile,
                "transitions": args.transitions,
                "stack_size": args.stack_size,
                "action_dim": args.action_dim,
                "observation_shape": list(observation_shape),
                "device": str(device),
                "threads": T.get_num_threads(),
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "seed": args.seed,
                "torch_version": T.__version__,
                "python_version": platform.python_version(),
            },
            "results": {name: result.to_dict() for name, result in results.items()},
            "metric_work": metric_work,
        }
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote benchmark report to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
