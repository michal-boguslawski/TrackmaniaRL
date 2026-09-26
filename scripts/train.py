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
from rl_lib.envs.make_env import make_env
from rl_lib.logger_setup import setup_logging, shutdown_logging
from rl_lib.networks.factory import Network
from rl_lib.run_config import RunConfig, load_config
from rl_lib.tracking.console_logger import ConsoleMetricsLogger
from rl_lib.tracking.mlflow_logger import MLflowLogger
from rl_lib.training.callbacks.checkpoints_save import CheckpointsSaveCallback
from rl_lib.training.callbacks.metrics_logger import MetricsLoggingCallback
from rl_lib.training.callbacks.params_logger import ParamsLoggingCallback
from rl_lib.training.callbacks.record_statistics import RecordStatisticLoggerCallback
from rl_lib.training.callbacks.record_video import RecordVideoCallback
from rl_lib.training.ppo_trainer import PPOTrainer
from rl_lib.training.rollout_collector import RolloutCollector


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")


for _signal in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_signal, _handle_termination)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)


def _resolve_device(config: RunConfig) -> T.device:
    if config.run.device == "auto":
        return T.device("cuda" if T.cuda.is_available() else "cpu")
    device = T.device(config.run.device)
    if device.type == "cuda" and not T.cuda.is_available():
        raise RuntimeError(f"Configured device {device} is unavailable")
    return device


def run_training(config: RunConfig, config_path: str | Path | None = None) -> None:
    level = getattr(logging, config.run.log_level)
    session_id = setup_logging(config.run.log_config, default_level=level) if config.run.log_config else setup_logging(default_level=level)
    logger = logging.getLogger(__name__)
    device = _resolve_device(config)
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
    with ExitStack() as stack:
        mlflow_logger = stack.enter_context(
            MLflowLogger(
                config.experiment_name,
                run_name=config.run_name,
                log_system_metrics=config.tracking.log_system_metrics,
            )
            if config.tracking.mlflow
            else nullcontext(None)
        )
        metrics_loggers = [item for item in (console_logger, mlflow_logger) if item is not None]
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

        trainer_callbacks = []
        if config.callbacks.checkpoints:
            trainer_callbacks.append(CheckpointsSaveCallback(agent, config.checkpoint_settings()))
        for metric_logger in metrics_loggers:
            if config.callbacks.metrics:
                trainer_callbacks.append(MetricsLoggingCallback(metric_logger, config.metrics_settings()))
        trainer = PPOTrainer(agent=agent, config=config.trainer, callbacks=trainer_callbacks)
        trainer.setup_train(config.run, config.rollout)

        collector_callbacks = []
        run_config_dump = config.model_dump(mode="json")
        for metric_logger in metrics_loggers:
            collector_callbacks.append(ParamsLoggingCallback(metric_logger, run_config=run_config_dump))
            if config.callbacks.episode_statistics:
                collector_callbacks.append(RecordStatisticLoggerCallback(
                    metric_logger, config.episode_statistics_settings()
                ))
        if config.callbacks.record_video:
            collector_callbacks.append(RecordVideoCallback(
                agent=video_agent,
                metrics_loggers=metrics_loggers,
                config=config.video_settings(),
            ))

        collector = RolloutCollector(
            env,
            buffer,
            trainer,
            config.rollout,
            callbacks=collector_callbacks,
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
    config = load_config(str(config_path))
    try:
        run_training(config, config_path=config_path)
    finally:
        shutdown_logging()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PPO agent using a YAML configuration")
    parser.add_argument("--config", default="configs/ppo_carracing.yaml", help="YAML run configuration")
    main(parser.parse_args().config)
