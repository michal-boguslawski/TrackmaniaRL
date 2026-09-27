"""Local checkpoint evaluation: load config + state dict, run inference.

Provides evaluate_checkpoint() for evaluating a saved model checkpoint
against the training configuration.
"""

from __future__ import annotations

from contextlib import closing
import logging
from pathlib import Path
from collections.abc import Mapping

import torch as T

from rl_lib.agent import Agent
from rl_lib.device import resolve_device
from rl_lib.envs.make_env import make_env
from rl_lib.evaluation.inference import log_evaluation_results, run_inference
from rl_lib.networks.factory import Network
from rl_lib.run_config import RunConfig, load_config
from rl_lib.tracking.base import MetricsLogger
from rl_lib.evaluation.runtime import _episode_stats_key, _default_metrics_loggers


logger = logging.getLogger(__name__)


def _load_local_network(network: Network, checkpoint_path: str | Path, device: T.device) -> Network:
    """Load either a state-dict checkpoint or the saved full local Network.

    Handles two checkpoint formats:
    1. Full Network object saved with T.save(network, path)
    2. State dict (or dict with "state_dict" key) saved with T.save(state_dict, path)

    Args:
        network: Empty Network instance to load state into.
        checkpoint_path: Path to checkpoint file.
        device: Target device for loading.

    Returns:
        Network with loaded weights on device.

    Raises:
        ValueError: If checkpoint format is unrecognized.
    """
    checkpoint = T.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, Network):
        return checkpoint.to(device)
    if isinstance(checkpoint, Mapping):
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain a model state dict")
        T.nn.Module.load_state_dict(network, state_dict)
        return network
    raise ValueError(f"Checkpoint {checkpoint_path} is neither a Network nor a state dict")


def _evaluation_environment(config: RunConfig, num_envs: int, record_video: bool):
    """Build evaluation environment from config with optional video recording."""
    environment = config.evaluation_environment(num_envs)
    if record_video:
        environment = environment.model_copy(update={
            "record_video": True,
            "video_folder": str(Path(config.video_folder) / "evaluation"),
            "vectorization_mode": "sync",
        })
    return environment


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    episodes: int = 5,
    record_video: bool = False,
    num_envs: int = 1,
    metrics_loggers: list[MetricsLogger] | None = None,
) -> list[float]:
    """Evaluate a local network state-dict checkpoint.

    Loads RunConfig from config_path, builds Network from config, loads
    checkpoint weights, and runs evaluation episodes.

    Args:
        config_path: Path to training config YAML.
        checkpoint_path: Path to model checkpoint (.pt file).
        episodes: Number of evaluation episodes.
        record_video: Whether to record videos.
        num_envs: Number of parallel environments.
        metrics_loggers: Tracking backends (defaults to console logger).

    Returns:
        List of episode returns.

    Raises:
        ValueError: If episodes < 1 or num_envs < 1.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")

    config = load_config(str(config_path))
    return _evaluate_configured_policy(
        config,
        episodes=episodes,
        num_envs=num_envs,
        metrics_loggers=_default_metrics_loggers(metrics_loggers),
        checkpoint_path=checkpoint_path,
        record_video=record_video,
        scope="local",
    )


def _evaluate_configured_policy(
    config: RunConfig,
    episodes: int,
    num_envs: int,
    metrics_loggers: list[MetricsLogger],
    record_video: bool,
    scope: str,
    checkpoint_path: str | Path | None = None,
    network: Network | None = None,
) -> list[float]:
    """Shared evaluation logic for local and MLflow checkpoints.

    Args:
        config: RunConfig for environment and agent settings.
        episodes: Number of episodes to run.
        num_envs: Parallel environments.
        metrics_loggers: Tracking backends.
        record_video: Whether to record videos.
        scope: Metric prefix scope ("local" or "mlflow").
        checkpoint_path: Optional path to local checkpoint.
        network: Optional pre-loaded Network (for MLflow models).

    Returns:
        List of episode returns.
    """
    device = resolve_device(config)
    env_config = _evaluation_environment(config, num_envs, record_video)

    with closing(make_env(env_config)) as env:
        if network is None:
            network = Network(
                observation_dim=env.observation_space.shape[-1],
                action_dim=env.action_space.shape[-1],
                stack_size=config.agent.stack_size,
                config=config.network,
            ).to(device)
        else:
            network = network.to(device)

        if checkpoint_path is not None:
            network = _load_local_network(network, checkpoint_path, device)
        agent = Agent(network, device, config.agent)

        results = run_inference(
            agent,
            env,
            episodes=episodes,
            seed=config.run.seed,
            temperature=config.agent.deterministic_temperature,
            episode_stats_key=_episode_stats_key(config.environment),
        )
        log_evaluation_results(results, metrics_loggers, step=0, scope=scope)

    if record_video:
        logger.info(
            "Evaluation videos saved to %s",
            Path(config.video_folder) / "evaluation",
        )
    return [result.return_ for result in results]
