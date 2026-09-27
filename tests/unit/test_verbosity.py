"""Unit coverage for metric verbosity filtering.

Verbosity is a property of what reaches a backend, not of what the trainer
computes, so the filtering lives in a `MetricsLogger` decorator: 0 forwards
nothing, 1 forwards only the core training losses, 2 forwards everything.
Parameters, configs, and artifacts are run metadata and stay unaffected.
"""

import pytest

from rl_lib.tracking.base import MetricsLogger
from rl_lib.tracking.verbosity import (
    Verbosity,
    VerbosityMetricsLogger,
    as_verbosity,
    filter_loggers,
    filter_metrics,
    is_core_loss,
)


CORE_LOSSES = {
    "loss/total": 1.0,
    "metrics/actor_loss": 0.5,
    "loss/critic": 0.25,
    "loss/entropy": 2.0,
}
DIAGNOSTIC_METRICS = {
    "rollout/returns": 12.0,
    "metrics/approx_kl": 0.01,
    "grad_norm/actor": 1.5,
    "training/lr_0": 1e-4,
    "episode/returns": 40.0,
    "evaluation/final/return_mean": 55.0,
}


class _RecordingLogger(MetricsLogger):
    """Backend that records every call it receives."""

    def __init__(self):
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.parameters: list[dict] = []
        self.configs: list[dict] = []
        self.state_dicts: list[str] = []
        self.artifacts: list[tuple[str, str | None]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((dict(metrics), step))

    def log_parameters(self, parameters: dict) -> None:
        self.parameters.append(dict(parameters))

    def log_config(self, config: dict, artifact_file: str = "config/run_config.yaml") -> None:
        self.configs.append(dict(config))

    def log_state_dict(self, state_dict: dict, artifact_path: str = "checkpoints") -> None:
        self.state_dicts.append(artifact_path)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))


# ---------------------------------------------------------------- policy


def test_verbosity_members_are_the_config_levels():
    assert [int(level) for level in Verbosity] == [0, 1, 2]


@pytest.mark.parametrize(
    ("verbosity", "core_losses", "diagnostics"),
    [
        (Verbosity.NONE, False, False),
        (Verbosity.CORE, True, False),
        (Verbosity.ALL, True, True),
    ],
)
def test_verbosity_says_what_is_produced_and_what_is_only_logged(
    verbosity, core_losses, diagnostics
):
    """The producers ask `logs_diagnostics` to skip work; `logs_core_losses`
    says whether the losses are worth their own synchronization."""
    assert verbosity.logs_core_losses is core_losses
    assert verbosity.logs_diagnostics is diagnostics


@pytest.mark.parametrize("value", [0, 1, 2])
def test_as_verbosity_accepts_the_config_level(value):
    assert as_verbosity(value) is Verbosity(value)


def test_as_verbosity_passes_a_member_through():
    assert as_verbosity(Verbosity.CORE) is Verbosity.CORE


def test_as_verbosity_rejects_an_unknown_level():
    with pytest.raises(ValueError, match=r"verbosity must be one of \[0, 1, 2\]"):
        as_verbosity(3)


# ---------------------------------------------------------------- predicates


def test_is_core_loss_covers_the_ppo_loss_family():
    assert is_core_loss("loss/total")
    assert is_core_loss("loss/critic")
    assert is_core_loss("loss/entropy")
    assert is_core_loss("metrics/actor_loss")


@pytest.mark.parametrize(
    "name",
    [
        "metrics/approx_kl",
        "rollout/explained_variance",
        "grad_norm/cnn",
        "training/entropy_coef",
        "episode/returns",
        "evaluation/periodic/return_mean",
    ],
)
def test_is_core_loss_rejects_diagnostics(name):
    assert not is_core_loss(name)


# ---------------------------------------------------------------- filtering


def test_verbosity_zero_filters_every_metric():
    assert filter_metrics({**CORE_LOSSES, **DIAGNOSTIC_METRICS}, 0) == {}


def test_verbosity_one_keeps_only_the_core_losses():
    assert filter_metrics({**CORE_LOSSES, **DIAGNOSTIC_METRICS}, 1) == CORE_LOSSES


def test_verbosity_two_keeps_everything():
    metrics = {**CORE_LOSSES, **DIAGNOSTIC_METRICS}

    assert filter_metrics(metrics, 2) == metrics


def test_filtering_does_not_alias_the_caller_dict():
    metrics = dict(CORE_LOSSES)

    selected = filter_metrics(metrics, 2)
    selected["loss/total"] = 99.0

    assert metrics["loss/total"] == 1.0


@pytest.mark.parametrize("verbosity", [3, -1, "all"])
def test_filter_metrics_rejects_an_unknown_verbosity(verbosity):
    with pytest.raises(ValueError, match="verbosity must be one of"):
        filter_metrics({}, verbosity)


# ---------------------------------------------------------------- decorator


def test_zero_verbosity_logger_never_calls_the_backend():
    logger = _RecordingLogger()

    VerbosityMetricsLogger(logger, 0).log_metrics({**CORE_LOSSES, **DIAGNOSTIC_METRICS}, step=1)

    assert logger.metrics == []


def test_core_loss_verbosity_drops_diagnostics_before_the_backend():
    logger = _RecordingLogger()

    VerbosityMetricsLogger(logger, 1).log_metrics({**CORE_LOSSES, **DIAGNOSTIC_METRICS}, step=4)

    assert logger.metrics == [(CORE_LOSSES, 4)]


def test_full_verbosity_logger_forwards_everything():
    logger = _RecordingLogger()
    metrics = {**CORE_LOSSES, **DIAGNOSTIC_METRICS}

    VerbosityMetricsLogger(logger, 2).log_metrics(metrics, step=9)

    assert logger.metrics == [(metrics, 9)]


def test_filtered_loggers_still_write_artifacts_and_parameters():
    logger = _RecordingLogger()
    filtered = VerbosityMetricsLogger(logger, 0)

    filtered.log_parameters({"lr": 1e-4})
    filtered.log_config({"run": {"seed": 1}})
    filtered.log_state_dict({"weight": 1}, artifact_path="checkpoints/step_200")
    filtered.log_artifact("/tmp/video.mp4", artifact_path="videos/step_000")

    assert logger.parameters == [{"lr": 1e-4}]
    assert logger.configs == [{"run": {"seed": 1}}]
    assert logger.state_dicts == ["checkpoints/step_200"]
    assert logger.artifacts == [("/tmp/video.mp4", "videos/step_000")]


def test_wrapper_rejects_an_unknown_verbosity():
    with pytest.raises(ValueError, match="verbosity must be one of"):
        VerbosityMetricsLogger(_RecordingLogger(), 5)


# ---------------------------------------------------------------- wiring


def test_filter_loggers_wraps_below_full_verbosity_only():
    logger = _RecordingLogger()

    assert filter_loggers([logger], 2) == [logger]
    assert isinstance(filter_loggers([logger], 1)[0], VerbosityMetricsLogger)


def test_filter_loggers_does_not_wrap_twice():
    filtered = filter_loggers([_RecordingLogger()], 1)

    assert filter_loggers(filtered, 0)[0] is filtered[0]
