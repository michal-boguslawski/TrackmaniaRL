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
    trainer: list[TrainingCallback] = field(default_factory=list)
    collector: list[CollectorCallback] = field(default_factory=list)


@dataclass(frozen=True)
class CallbackFactoryContext:
    config: RunConfig
    agent: Agent
    video_agent: Agent
    metrics_loggers: list[MetricsLogger]


CallbackBuilder = Callable[[CallbackFactoryContext], CallbackGroups]


def _checkpoints(context: CallbackFactoryContext) -> CallbackGroups:
    return CallbackGroups(
        trainer=[CheckpointsSaveCallback(context.agent, context.config.checkpoint_settings())]
    )


def _metrics(context: CallbackFactoryContext) -> CallbackGroups:
    return CallbackGroups(
        trainer=[
            MetricsLoggingCallback(metric_logger, context.config.metrics_settings())
            for metric_logger in context.metrics_loggers
        ]
    )


def _episode_statistics(context: CallbackFactoryContext) -> CallbackGroups:
    return CallbackGroups(
        collector=[
            RecordStatisticLoggerCallback(
                metric_logger, context.config.episode_statistics_settings()
            )
            for metric_logger in context.metrics_loggers
        ]
    )


def _record_video(context: CallbackFactoryContext) -> CallbackGroups:
    return CallbackGroups(
        collector=[RecordVideoCallback(
            agent=context.video_agent,
            metrics_loggers=context.metrics_loggers,
            config=context.config.video_settings(),
        )]
    )


def _evaluation(context: CallbackFactoryContext) -> CallbackGroups:
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
    """Create configured trainer/collector callbacks through a single registry."""
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
