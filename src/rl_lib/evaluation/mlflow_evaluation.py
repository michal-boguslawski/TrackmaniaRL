from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable

import yaml

from rl_lib.device import resolve_device
from rl_lib.evaluation.checkpoint import _evaluate_configured_policy
from rl_lib.networks.factory import Network
from rl_lib.run_config import RunConfig
from rl_lib.tracking.base import MetricsLogger
from rl_lib.evaluation.runtime import _default_metrics_loggers


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLflowModelRef:
    run_id: str
    run_name: str
    experiment_name: str
    unique_prefix: str
    start_time: int


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
    import mlflow

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
    import mlflow

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
    device = resolve_device(config)
    import mlflow

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
