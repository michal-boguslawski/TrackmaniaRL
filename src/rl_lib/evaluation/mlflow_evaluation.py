"""MLflow model evaluation: discover, select, and evaluate logged models.

Provides utilities to find MLflow runs with final_model artifacts,
resolve run identifiers, and evaluate loaded models with original config.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable

import mlflow
from mlflow.tracking import MlflowClient
import yaml

from rl_lib.device import resolve_device
from rl_lib.evaluation.checkpoint import _evaluate_configured_policy
from rl_lib.networks.factory import Network
from rl_lib.run_config import RunConfig
from rl_lib.tracking.base import MetricsLogger
from rl_lib.evaluation.runtime import _default_metrics_loggers


logger = logging.getLogger(__name__)

FINAL_MODEL_ARTIFACT = "final_model"


@dataclass(frozen=True)
class MLflowModelRef:
    """Reference to an MLflow run containing a final_model artifact.

    Attributes:
        run_id: Full MLflow run ID.
        run_name: Human-readable run name.
        experiment_name: Experiment name.
        unique_prefix: Shortest unique prefix of run_id for CLI selection.
        start_time: Run start timestamp (ms since epoch).
    """
    run_id: str
    run_name: str
    experiment_name: str
    unique_prefix: str
    start_time: int


def _shortest_unique_prefixes(run_ids: list[str]) -> dict[str, str]:
    """Compute shortest unique prefix for each run ID in a list.

    Args:
        run_ids: List of full run IDs.

    Returns:
        Dict mapping run_id -> shortest unique prefix.
    """
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


def _runs_with_logged_final_model(client: MlflowClient, experiment_id: str) -> set[str]:
    """Return run IDs in an experiment that logged a usable ``final_model``.

    MLflow 3 stores logged-model files outside the run's artifact directory
    (under ``<tracking_root>/models/m-<id>/artifacts``) and records the link
    between model and run in its logged-model registry. The model is therefore
    no longer visible when listing a run's artifacts, and the registry is the
    only reliable source of truth.

    Args:
        client: MLflow tracking client.
        experiment_id: Experiment to search.

    Returns:
        Set of run IDs with a usable ``final_model`` logged model. Empty when
        the installed MLflow has no logged-model registry.
    """
    search_logged_models = getattr(client, "search_logged_models", None)
    if search_logged_models is None:
        return set()

    run_ids = set()
    for model in search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"name = '{FINAL_MODEL_ARTIFACT}'",
    ):
        # A FAILED logged model has no loadable artifact behind it.
        if model.source_run_id and str(model.status) != "FAILED":
            run_ids.add(model.source_run_id)
    return run_ids


def _has_final_model_artifact(client: MlflowClient, run_id: str) -> bool:
    """Check whether a run's artifact directory contains ``final_model``.

    Only consulted for runs the logged-model registry does not cover, since
    older MLflow versions materialized the model inside the run's artifacts.

    Args:
        client: MLflow tracking client.
        run_id: MLflow run ID.

    Returns:
        True if the run has a ``final_model`` directory artifact.
    """
    return any(
        artifact.path == FINAL_MODEL_ARTIFACT and artifact.is_dir
        for artifact in client.list_artifacts(run_id)
    )


def discover_mlflow_models(experiment_name: str) -> list[MLflowModelRef]:
    """Return finished MLflow runs in an experiment that logged ``final_model``.

    Queries MLflow tracking server for runs in the experiment, filters for
    those with a ``final_model`` model logged, and computes unique prefixes
    for interactive selection.

    Args:
        experiment_name: MLflow experiment name.

    Returns:
        List of MLflowModelRef sorted by start_time (newest first).
    """

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("MLflow experiment %r was not found", experiment_name)
        return []

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    logged_model_runs = _runs_with_logged_final_model(client, experiment.experiment_id)
    available = []
    for run in runs:
        run_id = run.info.run_id
        if run_id not in logged_model_runs and not _has_final_model_artifact(client, run_id):
            continue
        available.append((
            run_id,
            run.data.tags.get("mlflow.runName", run_id),
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
    """Extract run ID from MLflow URI or prefix.

    Args:
        identifier: Run ID, "runs:/<run_id>/..." URI, or prefix.

    Returns:
        Run ID string.

    Raises:
        ValueError: If identifier is empty.
    """
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
    """Resolve a run ID/prefix, interactively listing models if none is given.

    Args:
        experiment_name: MLflow experiment to search.
        identifier: Run ID, prefix, or URI (optional; prompts if None).
        input_fn: Function for reading user input (default: built-in input).

    Returns:
        MLflowModelRef for the selected run.

    Raises:
        ValueError: If no models found, identifier ambiguous, or not found.
    """
    models = discover_mlflow_models(experiment_name)
    if not models:
        raise ValueError(
            f"No MLflow runs with a {FINAL_MODEL_ARTIFACT} artifact in {experiment_name!r}"
        )

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
    """Download and parse run_config.yaml from MLflow run artifacts.

    Handles backward compatibility by cleaning wrapper/callback configs
    to only include options supported by current wrapper versions.

    Args:
        run_id: MLflow run ID.

    Returns:
        Validated RunConfig.
    """

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

    # Runs are logged with unset fields omitted, so an absent wrapper list means
    # "use RunConfig defaults". Only rewrite lists that are actually present;
    # writing an empty list would drop the default wrappers the run trained with.
    environment = raw_config.get("environment", {})
    if "wrappers" in environment:
        environment["wrappers"] = [clean_wrapper(item) for item in environment["wrappers"]]
    callbacks = raw_config.get("callbacks", {})
    if isinstance(callbacks, dict):
        if "video_recording" in callbacks:
            callbacks["video_recording"] = clean_wrapper(callbacks["video_recording"])
        if "video_wrappers" in callbacks:
            callbacks["video_wrappers"] = [
                clean_wrapper(item) for item in callbacks["video_wrappers"]
            ]
    elif isinstance(callbacks, list):
        for callback in callbacks:
            if callback.get("name") == "record_video":
                if "recording" in callback:
                    callback["recording"] = clean_wrapper(callback["recording"])
                if "wrappers" in callback:
                    callback["wrappers"] = [
                        clean_wrapper(item) for item in callback["wrappers"]
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

    Args:
        run_id: MLflow run ID, URI, or prefix (None to list and prompt).
        experiment_name: Required if run_id not provided (to list models).
        episodes: Number of evaluation episodes.
        num_envs: Parallel environments.
        record_video: Whether to record videos.
        metrics_loggers: Tracking backends (defaults to console).
        input_fn: Input function for interactive selection.

    Returns:
        List of episode returns.

    Raises:
        ValueError: If episodes < 1, num_envs < 1, or experiment_name missing.
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

    network = mlflow.pytorch.load_model(
        f"runs:/{model_ref.run_id}/{FINAL_MODEL_ARTIFACT}",
        map_location=device,
    )
    if not isinstance(network, Network):
        raise TypeError(
            f"MLflow artifact {FINAL_MODEL_ARTIFACT} from run {model_ref.run_id} "
            "is not an rl_lib Network"
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


if __name__ == "__main__":
    evaluate_mlflow(experiment_name="Trackmania-RL-manual-smoke", episodes=1, num_envs=1, record_video=False)
