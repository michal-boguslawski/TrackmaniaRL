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
from importlib import resources
from pathlib import Path

import numpy as np
import torch as T

from rl_lib.agent import Agent
from rl_lib.buffers.rollout_buffer import RolloutBuffer
from rl_lib.device import resolve_device
from rl_lib.envs.make_env import make_env
from rl_lib.logger_setup import setup_logging, shutdown_logging
from rl_lib.networks.factory import Network
from rl_lib.profiling_config import ProfilingConfig, load_profiling_config
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


def run_training(
    config: RunConfig,
    config_path: str | Path | None = None,
    profiling_config: ProfilingConfig | None = None,
) -> None:
    """Execute full training run from validated config.

    Args:
        config: Validated RunConfig with all settings.
        config_path: Original config file path (for logging).
        profiling_config: Optional settings for a bounded PyTorch profiler trace.
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
    if device.type == "cuda":
        # Convolution shapes are static across updates, so cuDNN's autotuner
        # pays its search cost once per shape and caches the winner.
        T.backends.cudnn.benchmark = config.run.cudnn_benchmark
        if config.run.clear_cuda_cache:
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
        if device.type == "cuda" and config.run.channels_last:
            # The CNN's Conv2d layers are the only 2D operators in the network.
            # NHWC weights match cuDNN's native kernel layout and the
            # channels_last strides of the permuted NHWC observations, so the
            # per-convolution NCHW<->NHWC conversion kernels disappear.
            network.cnn.to(memory_format=T.channels_last)
        if config.run.torch_compile:
            network = T.compile(network, mode=config.run.torch_compile_mode)
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
            if profiling_config is None or not profiling_config.enabled:
                collector.run(config.run.total_steps)
            else:
                activities = [T.profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(T.profiler.ProfilerActivity.CUDA)
                trace_dir = Path(profiling_config.output_dir) / session_id
                wait_steps, warmup_steps = profiling_config.resolve_schedule(
                    config.rollout.buffer_size
                )
                profiler = T.profiler.profile(
                    activities=activities,
                    schedule=T.profiler.schedule(
                        wait=wait_steps,
                        warmup=warmup_steps,
                        active=profiling_config.active_steps,
                        repeat=profiling_config.repeat,
                    ),
                    on_trace_ready=T.profiler.tensorboard_trace_handler(str(trace_dir)),
                    record_shapes=profiling_config.record_shapes,
                    profile_memory=profiling_config.profile_memory,
                    with_stack=profiling_config.with_stack,
                )
                with profiler:
                    collector.run(config.run.total_steps, profiler=profiler)
                logger.info("Profiler traces written to %s", trace_dir)
        except KeyboardInterrupt:
            logger.warning("Rollout interrupted; stopping training")
            raise
        except Exception:
            logger.exception("rollout_collector.run failed")
            raise


def main(
    config_path: str | Path,
    profiling_config_path: str | Path | None = None,
    profile: bool | None = None,
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
    default_profiling_config = resources.files("rl_lib.config") / "profiling.yaml"
    profiling_path = profiling_config_path or default_profiling_config
    profiling_settings = load_profiling_config(profiling_path)
    profiling_enabled = profiling_settings.enabled if profile is None else profile
    profiling_settings = profiling_settings.model_copy(update={"enabled": profiling_enabled})
    try:
        run_training(
            config,
            config_path=config_path,
            profiling_config=profiling_settings,
        )
    finally:
        shutdown_logging()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PPO agent using a YAML configuration")
    parser.add_argument("--config", default="configs/ppo_carracing.yaml", help="YAML run configuration")
    parser.add_argument("--profiling-config", help="Separate YAML configuration for optional PyTorch profiling")
    parser.add_argument("--profile", dest="profile", action="store_true", default=None)
    parser.add_argument("--no-profile", dest="profile", action="store_false")
    args = parser.parse_args()
    main(args.config, profiling_config_path=args.profiling_config, profile=args.profile)
