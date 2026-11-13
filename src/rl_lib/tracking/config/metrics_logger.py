from typing import Literal

from pydantic import Field

from rl_lib.config_base import StrictModel
from rl_lib.tracking.config.callbacks.episode_statistics import (
    EpisodeStatisticsCallbackSettings,
)


class TrackingSettings(StrictModel):
    """Which scalars reach the tracking backends.

    Attributes:
        console: Enable the console metrics logger.
        mlflow: Enable the MLflow tracking backend.
        local_artifacts: Enable local artifact logging. Artifacts are stored in logs folder.
        experiment_name: MLflow experiment (defaults to the environment id).
        run_name: MLflow run name (defaults to "<run name>/<session id>").
        log_system_metrics: Let MLflow sample host metrics; only honoured at
            ``verbosity`` 2, since system metrics are diagnostics.
        verbosity: Metric detail level: 0 logs no metrics, 1 logs only the
            core training losses, 2 logs every metric.
    """

    console: bool = True
    mlflow: bool = True
    local_artifacts: bool = False
    experiment_name: str | None = None
    run_name: str | None = None
    log_system_metrics: bool = True
    verbosity: Literal[0, 1, 2] = Field(
        default=2,
        description="Metric detail: 0 none, 1 core training losses only, 2 everything.",
    )
    callbacks: list[EpisodeStatisticsCallbackSettings] = Field(default_factory=list)
