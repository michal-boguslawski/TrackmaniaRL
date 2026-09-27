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
from contextlib import ExitStack, closing, nullcontext
import gc
import logging
import random
import signal
from pathlib import Path

import numpy as np
import torch as T

from rl_lib.agent import Agent
from rl_lib.buffers.rollout_buffer import RolloutBuffer
from rl_lib.device import resolve_device
from rl_lib.envs.make_env import make_env
from rl_lib.logger_setup import setup_logging, shutdown_logging
from rl_lib.networks.factory import Network
from rl_lib.run_config import RunConfig, load_config
from rl_lib.tracking.console_logger import ConsoleMetricsLogger
from rl_lib.tracking.local_artifact_logger import LocalArtifactLogger
from rl_lib.tracking.mlflow_logger import MLflowLogger
from rl_lib.tracking.verbosity import Verbosity, filter_loggers
from rl_lib.training.callbacks.factory import create_callbacks
from rl_lib.training.factory import create_trainer
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


def run_training(config: RunConfig, config_path: str | Path | None = None) -> None:
    """Execute full training run from validated config.

    Args:
        config: Validated RunConfig with all settings.
        config_path: Original config file path (for logging).
    """
    level = getattr(logging, config.run.log_level)
    session_id = setup_logging(config.run.log_config, default_level=level) if config.run.log_config else setup_logging(default_level=level)
    logger = logging.getLogger(__name__)
    device = resolve_device(config)
    config = config.with_runtime(
        session_id=session_id,
        config_path=str(config_path) if config_path is not None else None,
        device=str(device),
        mlflow_experiment_name=config.experiment_name if config.tracking.mlflow else None,
        mlflow_run_name=config.run_name if config.tracking.mlflow else None,
    )
    _seed_everything(config.run.seed)
    gc.collect()
    if device.type == "cuda" and config.run.clear_cuda_cache:
        T.cuda.empty_cache()
        T.cuda.reset_peak_memory_stats(device)

    console_logger = ConsoleMetricsLogger() if config.tracking.console else None
    verbosity = config.tracking.verbosity
    with ExitStack() as stack:
        mlflow_logger = stack.enter_context(
            MLflowLogger(
                config.experiment_name,
                run_name=config.run_name,
                # Host metrics are diagnostics, so they only belong at the
                # verbosity that logs every metric.
                log_system_metrics=(
                    config.tracking.log_system_metrics and verbosity == Verbosity.ALL
                ),
            )
            if config.tracking.mlflow
            else nullcontext(None)
        )
        local_artifact_logger = LocalArtifactLogger(
            Path("logs") / "artifacts" / session_id
        )
        metrics_loggers = filter_loggers(
            [
                item
                for item in (console_logger, mlflow_logger, local_artifact_logger)
                if item is not None
            ],
            verbosity,
        )
        env = stack.enter_context(closing(make_env(config.environment)))
        config = config.with_runtime(
            observation_shape=list(env.observation_space.shape),
            action_shape=list(env.action_space.shape),
        )

        network = Network(
            observation_dim=config.runtime.observation_shape[-1],
            action_dim=config.runtime.action_shape[-1],
            stack_size=config.agent.stack_size,
            config=config.network,
        ).to(device)
        agent = Agent(network, device, config.agent)
        video_agent = Agent(network, device, config.agent)
        buffer = RolloutBuffer(config.rollout, stack_size=config.agent.stack_size)

        run_config_dump = config.model_dump(mode="json", exclude_unset=True)
        callback_groups = create_callbacks(
            config=config,
            agent=agent,
            video_agent=video_agent,
            metrics_loggers=metrics_loggers,
            run_config=run_config_dump,
        )

        trainer = create_trainer(
            config.run.algorithm,
            agent=agent,
            config=config.trainer,
            callbacks=callback_groups.trainer,
            verbosity=verbosity,
        )
        trainer.setup_train(config.run, config.rollout)

        collector = RolloutCollector(
            env,
            buffer,
            trainer,
            config.rollout,
            callbacks=callback_groups.collector,
            seed=config.run.seed,
            run_config=run_config_dump,
        )
        logger.info("Starting training with device=%s config=%s", device, config_path or "<object>")
        try:
            collector.run(config.run.total_steps)
        except KeyboardInterrupt:
            logger.warning("Rollout interrupted; stopping training")
            raise
        except Exception:
            logger.exception("rollout_collector.run failed")
            raise


def main(config_path: str | Path) -> None:
    """CLI entrypoint: load config and run training."""
    config = load_config(str(config_path))
    try:
        run_training(config, config_path=config_path)
    finally:
        shutdown_logging()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PPO agent using a YAML configuration")
    parser.add_argument("--config", default="configs/ppo_carracing.yaml", help="YAML run configuration")
    main(parser.parse_args().config)
