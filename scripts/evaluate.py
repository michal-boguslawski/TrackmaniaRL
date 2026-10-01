"""Evaluate a local checkpoint or an MLflow model/checkpoint from the CLI.

Examples
--------
Evaluate a selected MLflow checkpoint::

    uv run --extra gpu python scripts/evaluate.py --run-id RUN_ID \\
        --checkpoint-step 2200000 --episodes 10

Evaluate the final MLflow model::

    uv run --extra gpu python scripts/evaluate.py --run-id RUN_ID
"""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from rl_lib.evaluation import evaluate_checkpoint, evaluate_mlflow, summarize_returns


def _parser() -> argparse.ArgumentParser:
    """Build the standalone evaluation command-line parser."""
    parser = argparse.ArgumentParser(description="Evaluate a trained policy.")
    parser.add_argument(
        "--source",
        choices=("mlflow", "local"),
        default="mlflow",
        help="Model source (default: mlflow).",
    )
    parser.add_argument("--run-id", help="MLflow run ID, prefix, or runs:/ URI.")
    parser.add_argument(
        "--experiment",
        default="CarRacing-v3",
        help="MLflow experiment name (default: CarRacing-v3).",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        help="Evaluate this periodic MLflow checkpoint step instead of final_model.",
    )
    parser.add_argument("--config", type=Path, help="Training YAML for local checkpoints.")
    parser.add_argument("--checkpoint", type=Path, help="Local checkpoint file.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to evaluate (default: 5).")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel evaluation environments.")
    parser.add_argument("--record-video", action="store_true", help="Record evaluation videos.")
    return parser


def main() -> None:
    """Parse arguments and evaluate the requested policy."""
    args = _parser().parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be at least 1")

    if args.source == "mlflow":
        if args.config is not None or args.checkpoint is not None:
            raise SystemExit("--config and --checkpoint apply only with --source local")
        returns = evaluate_mlflow(
            run_id=args.run_id,
            experiment_name=args.experiment,
            checkpoint_step=args.checkpoint_step,
            episodes=args.episodes,
            num_envs=args.num_envs,
            record_video=args.record_video,
        )
    else:
        if args.run_id is not None or args.checkpoint_step is not None:
            raise SystemExit("--run-id and --checkpoint-step apply only with --source mlflow")
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required with --source local")
        config_path = args.config or Path(
            str(resources.files("rl_lib.config") / "ppo_carracing.yaml")
        )
        returns = evaluate_checkpoint(
            config_path,
            args.checkpoint,
            episodes=args.episodes,
            num_envs=args.num_envs,
            record_video=args.record_video,
        )

    summary = summarize_returns(returns)
    print(f"Episodes: {int(summary['episodes'])}")
    print(f"Return mean ± std: {summary['mean']:.3f} ± {summary['std']:.3f}")
    print(
        "Return min / median / max: "
        f"{summary['min']:.3f} / {summary['median']:.3f} / {summary['max']:.3f}"
    )


if __name__ == "__main__":
    main()
