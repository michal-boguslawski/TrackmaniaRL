from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable
from collections.abc import Mapping

import torch as T
import yaml

from rl_lib.agent import Agent
from rl_lib.envs.make_env import make_env
from rl_lib.inference import EpisodeResult, log_evaluation_results, run_inference
from rl_lib.networks.factory import Network
from rl_lib.run_config import RunConfig, load_config
from rl_lib.tracking.base import MetricsLogger


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLflowModelRef:
    run_id: str
    run_name: str
    experiment_name: str
    unique_prefix: str
    start_time: int


def _resolve_device(config: RunConfig) -> T.device:
    if config.run.device == "auto":
        return T.device("cuda" if T.cuda.is_available() else "cpu")
    device = T.device(config.run.device)
    if device.type == "cuda" and not T.cuda.is_available():
        raise RuntimeError(f"Configured device {device} is unavailable")
    return device


def _evaluation_environment(config: RunConfig, num_envs: int, record_video: bool):
    environment = config.evaluation_environment(num_envs)
    if record_video:
        environment = environment.model_copy(update={
            "record_video": True,
            "video_folder": str(Path(config.video_folder) / "evaluation"),
            "vectorization_mode": "sync",
        })
    return environment


def _episode_stats_key(config: RunConfig) -> str:
    return next(
        (
            wrapper.stats_key
            for wrapper in config.environment.wrappers
            if wrapper.name == "record_episode_stats"
        ),
        "episode",
    )


def _default_metrics_loggers(metrics_loggers: list[MetricsLogger] | None) -> list[MetricsLogger]:
    if metrics_loggers is not None:
        return metrics_loggers
    from rl_lib.tracking.console_logger import ConsoleMetricsLogger

    return [ConsoleMetricsLogger()]


def _load_local_network(network: Network, checkpoint_path: str | Path, device: T.device) -> Network:
    """Load either a state-dict checkpoint or the saved full local Network."""
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


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    episodes: int = 5,
    record_video: bool = False,
    num_envs: int = 1,
    metrics_loggers: list[MetricsLogger] | None = None,
) -> list[float]:
    """Evaluate a local network state-dict checkpoint."""
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
    device = _resolve_device(config)
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
            episode_stats_key=_episode_stats_key(config),
        )
        log_evaluation_results(results, metrics_loggers, step=0, scope=scope)

    if record_video:
        logger.info(
            "Evaluation videos saved to %s",
            Path(config.video_folder) / "evaluation",
        )
    return [result.return_ for result in results]


def _mlflow_module():
    import mlflow

    return mlflow


def _shortest_unique_prefixes(run_ids: list[str]) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    for run_id in run_ids:
        for size in range(1, len(run_id) + 1):
            candidate = run_id[:size]
            if sum(other.startswith(candidate) for other in run_ids) == 1:
                prefixes[run_id] = candidate
                break
        else:
            prefixes[run_id] = run_id
    return prefixes


def discover_mlflow_models(experiment_name: str) -> list[MLflowModelRef]:
    """Return finished MLflow runs in an experiment that logged ``final_model``."""
    mlflow = _mlflow_module()
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("MLflow experiment %r was not found", experiment_name)
        return []

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    available = []
    for run in runs:
        if not any(
            artifact.path == "final_model" and artifact.is_dir
            for artifact in client.list_artifacts(run.info.run_id)
        ):
            continue
        available.append((
            run.info.run_id,
            run.data.tags.get("mlflow.runName", run.info.run_id),
            run.info.start_time or 0,
        ))

    prefixes = _shortest_unique_prefixes([entry[0] for entry in available])
    return [
        MLflowModelRef(
            run_id=run_id,
            run_name=run_name,
            experiment_name=experiment_name,
            unique_prefix=prefixes[run_id],
            start_time=start_time,
        )
        for run_id, run_name, start_time in sorted(available, key=lambda item: item[2], reverse=True)
    ]


def _run_id_from_identifier(identifier: str) -> str:
    value = identifier.strip()
    if value.startswith("runs:/"):
        value = value.removeprefix("runs:/").split("/", 1)[0]
    if not value:
        raise ValueError("MLflow run identifier cannot be empty")
    return value


def select_mlflow_run(
    experiment_name: str,
    identifier: str | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> MLflowModelRef:
    """Resolve a run ID/prefix, interactively listing models if none is given."""
    models = discover_mlflow_models(experiment_name)
    if not models:
        raise ValueError(f"No MLflow runs with a final_model artifact in {experiment_name!r}")

    if identifier is None:
        logger.info("Available MLflow models in experiment %s:", experiment_name)
        for model in models:
            logger.info("  %s  %s  (run_id=%s)", model.unique_prefix, model.run_name, model.run_id)
        prompt = input_fn or input
        identifier = prompt("Select a model by its unique run-ID prefix: ").strip()

    requested = _run_id_from_identifier(identifier)
    matches = [model for model in models if model.run_id.startswith(requested)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = ", ".join(model.unique_prefix for model in matches)
        raise ValueError(f"MLflow run prefix {requested!r} is ambiguous; use one of: {options}")
    raise ValueError(f"No model run in {experiment_name!r} matches prefix {requested!r}")


def _load_mlflow_config(run_id: str) -> RunConfig:
    mlflow = _mlflow_module()
    config_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="config/run_config.yaml",
    )
    with Path(config_path).open("rt", encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream)
    if not isinstance(raw_config, dict):
        raise ValueError(f"MLflow run {run_id} has no valid config/run_config.yaml artifact")
    # Training logs resolved process facts as well; they are not authored config.
    raw_config.pop("runtime", None)
    # Older runs serialized every wrapper default. Keep only options supported
    # by each wrapper before strict validation, preserving older MLflow runs.
    wrapper_options = {
        "grayscale": {"keep_dim"},
        "frame_stack": {"stack_size", "frame_padding_type"},
        "record_episode_stats": {"stats_buffer_length", "stats_key"},
        "record_video": {
            "video_folder", "episode_trigger", "episode_interval", "video_length",
            "video_name_prefix", "video_fps", "video_disable_logger",
        },
        "max_and_skip": {"skip"},
        "reward_on_done": {"on_terminated", "on_truncated", "on_win"},
    }

    def clean_wrapper(raw_wrapper: dict) -> dict:
        name = raw_wrapper.get("name")
        allowed = wrapper_options.get(name, set())
        return {key: value for key, value in raw_wrapper.items() if key == "name" or key in allowed}

    environment = raw_config.get("environment", {})
    environment["wrappers"] = [clean_wrapper(item) for item in environment.get("wrappers", [])]
    callbacks = raw_config.get("callbacks", {})
    if isinstance(callbacks, dict):
        if "video_recording" in callbacks:
            callbacks["video_recording"] = clean_wrapper(callbacks["video_recording"])
        callbacks["video_wrappers"] = [
            clean_wrapper(item) for item in callbacks.get("video_wrappers", [])
        ]
    elif isinstance(callbacks, list):
        for callback in callbacks:
            if callback.get("name") == "record_video":
                if "recording" in callback:
                    callback["recording"] = clean_wrapper(callback["recording"])
                callback["wrappers"] = [
                    clean_wrapper(item) for item in callback.get("wrappers", [])
                ]
    return RunConfig.model_validate(raw_config)


def evaluate_mlflow(
    run_id: str | None = None,
    experiment_name: str | None = None,
    episodes: int = 5,
    num_envs: int = 1,
    record_video: bool = False,
    metrics_loggers: list[MetricsLogger] | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> list[float]:
    """Evaluate the ``final_model`` artifact for an MLflow run.

    ``run_id`` may also be a ``runs:/<run_id>/...`` URI. With no identifier,
    the available model artifacts in ``experiment_name`` are listed and a
    unique run-ID prefix is requested.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")

    if run_id is None or run_id.strip() == "":
        if not experiment_name:
            raise ValueError("experiment_name is required to list MLflow models")
        model_ref = select_mlflow_run(experiment_name, input_fn=input_fn)
    elif experiment_name is not None:
        model_ref = select_mlflow_run(experiment_name, run_id)
    else:
        resolved_run_id = _run_id_from_identifier(run_id)
        model_ref = MLflowModelRef(
            run_id=resolved_run_id,
            run_name=resolved_run_id,
            experiment_name="",
            unique_prefix=resolved_run_id,
            start_time=0,
        )

    config = _load_mlflow_config(model_ref.run_id)
    device = _resolve_device(config)
    mlflow = _mlflow_module()
    network = mlflow.pytorch.load_model(
        f"runs:/{model_ref.run_id}/final_model",
        map_location=device,
    )
    if not isinstance(network, Network):
        raise TypeError(
            f"MLflow artifact final_model from run {model_ref.run_id} is not an rl_lib Network"
        )

    return _evaluate_configured_policy(
        config,
        episodes=episodes,
        num_envs=num_envs,
        metrics_loggers=_default_metrics_loggers(metrics_loggers),
        record_video=record_video,
        scope="mlflow",
        network=network,
    )
