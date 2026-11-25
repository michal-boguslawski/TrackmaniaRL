"""Main training pipeline: config loading, environment setup, training loop.

Orchestrates the full PPO training run:
1. Setup logging (console + MLflow)
2. Resolve device (CUDA/CPU)
3. Seed RNGs
4. Create vectorized environment with wrappers
5. Build Network (CNN -> Temporal -> Actor/Critic)
6. Create Agent, RolloutBuffer, PPOTrainer, RolloutCollector
7. Register callbacks (checkpoints, metrics, video, evaluation, etc.)
8. Run collector loop (environment steps + PPO updates)
9. Cleanup logging
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import signal
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np
import torch as T

from rl_lib.device import resolve_device
from rl_lib.envs.make_env import make_env
from rl_lib.logger_setup import resolve_session_id, setup_logging, shutdown_logging
from rl_lib.networks.utils import flatten_dict
from rl_lib.paths import RunPaths
from rl_lib.run_config import RunConfig, load_config
from rl_lib.tracking.builder import build_callbacks, build_metrics_logger
from rl_lib.training.base import BuildContext
from rl_lib.training.ppo.builder import build_ppo
from rl_lib.training.rollout_collector import RolloutCollector


def _handle_termination(signum, frame):
    """Signal handler: convert SIGTERM/SIGINT to KeyboardInterrupt."""
    raise KeyboardInterrupt(f"received signal {signum}")


for _signal in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_signal, _handle_termination)


def _seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)


def run_training(
    config: RunConfig,
    config_path: str | Path | None = None,
) -> None:
    """Execute full training run from validated config.

    Args:
        config: Validated RunConfig with all settings.
        config_path: Original config file path (for logging).
        profiling_config: Optional settings for a bounded PyTorch profiler trace.
    """
    level = getattr(logging, config.run.log_level)
    session_id = resolve_session_id()
    run_paths = RunPaths(
        Path(config.run.logs_root),
        session_id,
        experiment=config.tracking.experiment_name or config.environment.id,
        algorithm=config.algorithm.name,
    )
    (
        setup_logging(
            session_id,
            run_paths.log_file,
            config.run.log_config,
            default_level=level,
        )
        if config.run.log_config
        else setup_logging(
            session_id,
            run_paths.log_file,
            default_level=level,
        )
    )
    logger = logging.getLogger(__name__)
    device = resolve_device(config)
    _seed_everything(config.run.seed)
    gc.collect()
    # if device.type == "cuda":
    # Convolution shapes are static across updates, so cuDNN's autotuner
    # pays its search cost once per shape and caches the winner.
    # T.backends.cudnn.benchmark = config.run.cudnn_benchmark
    # if config.run.clear_cuda_cache:
    #     T.cuda.empty_cache()
    #     T.cuda.reset_peak_memory_stats(device)

    verbosity = config.tracking.verbosity
    with ExitStack() as stack:
        env = stack.enter_context(closing(make_env(config.environment, device=device)))
        build_ctx = BuildContext(
            config=config,
            device=device,
            verbosity=verbosity,
            obs_shape=env.observation_space.shape,
            action_space=env.single_action_space,
        )
        metrics_logger = build_metrics_logger(
            config.tracking,
            run_paths=run_paths,
        )
        policy, trainer = build_ppo(build_ctx)
        callbacks = build_callbacks(
            metrics_logger,
            config.tracking.callbacks,
            base_env=config.environment,
            run_paths=run_paths,
            device=device,
            run_seed=config.run.seed,
        )
        trainer.set_callbacks(callbacks)

        metrics_logger.log_config(config.model_dump(mode="json"))
        metrics_logger.log_parameters(
            flatten_dict(
                config.model_dump(mode="json", exclude={"algorithm": {"network"}})
            )
        )
        metrics_logger.log_parameters(trainer.modules.config())

        collector = RolloutCollector(
            env,
            policy,
            trainer,
            callbacks=callbacks,
            seed=config.run.seed,
        )
        logger.info(
            "Starting training with device=%s config=%s",
            device,
            config_path or "<object>",
        )
        try:
            collector.run(config.run.total_steps)
        except KeyboardInterrupt:
            logger.warning("Rollout interrupted; stopping training")
            raise
        except Exception:
            logger.exception("rollout_collector.run failed")
            raise


def main(
    config_path: str | Path,
) -> None:
    """CLI entrypoint: load run and profiling configs, then train.

    Args:
        config_path: YAML training configuration.
        profiling_config_path: Separate YAML profiling configuration. Defaults
            to the packaged ``profiling.yaml``.
        profile: Optional CLI override for the profiling config's ``enabled``
            setting. ``None`` follows the config value.
    """
    config = load_config(str(config_path))
    try:
        run_training(
            config,
            config_path=config_path,
        )
    finally:
        shutdown_logging()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the PPO agent using a YAML configuration"
    )
    parser.add_argument(
        "--config", default="config/new.yaml", help="YAML run configuration"
    )
    args = parser.parse_args()
    main(args.config)
