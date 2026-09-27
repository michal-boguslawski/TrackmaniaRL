"""Callback factory: builds TrainingCallback/CollectorCallback instances from config.

Registers callback constructors in CALLBACK_FACTORIES and provides create_callbacks()
to instantiate all configured callbacks with resolved settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rl_lib.agent import Agent
from rl_lib.run_config import RunConfig
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.base import CollectorCallback, TrainingCallback
from rl_lib.training.callbacks.checkpoints_save import CheckpointsSaveCallback
from rl_lib.training.callbacks.evaluate import EvaluationCallback
from rl_lib.training.callbacks.final_model_save import FinalModelSaveCallback
from rl_lib.training.callbacks.metrics_logger import MetricsLoggingCallback
from rl_lib.training.callbacks.params_logger import ParamsLoggingCallback
from rl_lib.training.callbacks.record_statistics import RecordStatisticLoggerCallback
from rl_lib.training.callbacks.record_video import RecordVideoCallback


@dataclass
class CallbackGroups:
    """Container for callbacks split by when they're invoked.

    Attributes:
        trainer: Callbacks invoked during PPO optimization (TrainingCallback).
        collector: Callbacks invoked during environment rollout (CollectorCallback).
    """
    trainer: list[TrainingCallback] = field(default_factory=list)
    collector: list[CollectorCallback] = field(default_factory=list)


@dataclass(frozen=True)
class CallbackFactoryContext:
    """Resolved dependencies passed to each callback builder.

    Attributes:
        config: Full RunConfig (provides access to all settings).
        agent: Training agent (for checkpoints, evaluation).
        video_agent: Evaluation agent with video recording setup.
        metrics_loggers: List of MetricsLogger backends (console, MLflow).
    """
    config: RunConfig
    agent: Agent
    video_agent: Agent
    metrics_loggers: list[MetricsLogger]


CallbackBuilder = Callable[[CallbackFactoryContext], CallbackGroups]


def _checkpoints(context: CallbackFactoryContext) -> CallbackGroups:
    """Build checkpoint saving callback (trainer-level)."""
    return CallbackGroups(
        trainer=[CheckpointsSaveCallback(context.agent, context.config.checkpoint_settings())]
    )


def _metrics(context: CallbackFactoryContext) -> CallbackGroups:
    """Build metrics logging callback (trainer-level)."""
    return CallbackGroups(
        trainer=[
            MetricsLoggingCallback(metric_logger, context.config.metrics_settings())
            for metric_logger in context.metrics_loggers
        ]
    )


def _episode_statistics(context: CallbackFactoryContext) -> CallbackGroups:
    """Build episode statistics logging callback (collector-level)."""
    return CallbackGroups(
        collector=[
            RecordStatisticLoggerCallback(
                metric_logger, context.config.episode_statistics_settings()
            )
            for metric_logger in context.metrics_loggers
        ]
    )


def _record_video(context: CallbackFactoryContext) -> CallbackGroups:
    """Build video recording callback (collector-level)."""
    return CallbackGroups(
        collector=[RecordVideoCallback(
            agent=context.video_agent,
            metrics_loggers=context.metrics_loggers,
            config=context.config.video_settings(),
        )]
    )


def _evaluation(context: CallbackFactoryContext) -> CallbackGroups:
    """Build periodic evaluation callback (collector-level)."""
    return CallbackGroups(
        collector=[EvaluationCallback(
            agent=context.agent,
            metrics_loggers=context.metrics_loggers,
            config=context.config.evaluation_settings(),
        )]
    )


CALLBACK_FACTORIES: dict[str, CallbackBuilder] = {
    "checkpoints": _checkpoints,
    "metrics": _metrics,
    "episode_statistics": _episode_statistics,
    "record_video": _record_video,
    "evaluation": _evaluation,
}


def create_callbacks(
    config: RunConfig,
    agent: Agent,
    video_agent: Agent,
    metrics_loggers: list[MetricsLogger],
    run_config: dict | None = None,
) -> CallbackGroups:
    """Create all configured trainer/collector callbacks.

    Always adds ParamsLoggingCallback (logs full config at start) and
    FinalModelSaveCallback (saves final model at end) in addition to
    callbacks specified in config.callbacks.

    Args:
        config: Validated RunConfig with callbacks list.
        agent: Training agent.
        video_agent: Evaluation agent for video recording.
        metrics_loggers: MetricsLogger backends (console, MLflow).
        run_config: Full config dict for MLflow param logging.

    Returns:
        CallbackGroups with populated trainer and collector lists.
    """
    context = CallbackFactoryContext(
        config=config,
        agent=agent,
        video_agent=video_agent,
        metrics_loggers=metrics_loggers,
    )
    groups = CallbackGroups(
        collector=[
            ParamsLoggingCallback(metric_logger, run_config=run_config)
            for metric_logger in metrics_loggers
        ]
    )

    for callback_config in config.callbacks:
        created = CALLBACK_FACTORIES[callback_config.name](context)
        groups.trainer.extend(created.trainer)
        groups.collector.extend(created.collector)

    groups.collector.append(FinalModelSaveCallback(
        agent=agent,
        folder=config.checkpoint_folder,
        metrics_loggers=metrics_loggers,
    ))
    return groups
