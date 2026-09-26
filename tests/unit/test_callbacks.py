"""Unit tests for the training/collector callback layer.

Covers the abstract hook contracts, the callback dispatcher, the console
metrics backend and each concrete callback. Known source defects are pinned
with strict xfails instead of being papered over.
"""

from __future__ import annotations

import logging
import types
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch as T

from rl_lib.agent import Agent
from rl_lib.training.callbacks.base import (
    CallbackList,
    CollectorCallback,
    TrainingCallback,
)
from rl_lib.training.callbacks.checkpoints_save import CheckpointsSaveCallback
from rl_lib.training.callbacks.metrics_logger import MetricsLoggingCallback
from rl_lib.training.callbacks.params_logger import ParamsLoggingCallback
from rl_lib.training.callbacks.record_statistics import RecordStatisticLoggerCallback
from rl_lib.training.callbacks.record_video import RecordVideoCallback
from rl_lib.training.callbacks.utils import stop_video_recording
from rl_lib.tracking.base import MetricsLogger


class FakeLogger(MetricsLogger):
    def __init__(self):
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.parameters: list[dict] = []
        self.artifacts: list[tuple[str, str | None]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((dict(metrics), step))

    def log_parameters(self, parameters: dict) -> None:
        self.parameters.append(dict(parameters))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))


class MinimalTrainingCallback(TrainingCallback):
    def on_start(self, *args, **kwargs) -> None: ...
    def on_minibatch(self, *args, **kwargs) -> None: ...
    def on_end(self, *args, **kwargs) -> None: ...
    def on_epoch(self, *args, **kwargs) -> None: ...


class MinimalCollectorCallback(CollectorCallback):
    def on_rollout_start(self, *args, **kwargs) -> None: ...
    def on_env_step(self, *args, **kwargs) -> None: ...
    def on_rollout_end(self, *args, **kwargs) -> None: ...
    def flush(self, *args, **kwargs) -> None: ...


# ---------------------------------------------------------------- contracts


def test_training_cannot_be_instantiated_without_every_hook():
    with pytest.raises(TypeError):
        TrainingCallback()  # type: ignore[abstract]


def test_collector_cannot_be_instantiated_without_every_hook():
    with pytest.raises(TypeError):
        CollectorCallback()  # type: ignore[abstract]


def test_a_complete_training_subclass_is_usable():
    assert isinstance(MinimalTrainingCallback(), TrainingCallback)


def test_a_complete_collector_subclass_is_usable():
    assert isinstance(MinimalCollectorCallback(), CollectorCallback)


def test_a_missing_hook_keeps_the_subclass_abstract():
    class _Partial(TrainingCallback):
        def on_start(self, *args, **kwargs) -> None: ...

    with pytest.raises(TypeError):
        _Partial()  # type: ignore[abstract]


# ---------------------------------------------------------------- dispatch


def test_callback_list_dispatches_to_every_callback_in_order():
    calls: list[tuple[str, str, dict]] = []

    class _Recording(MinimalTrainingCallback):
        def __init__(self, name: str):
            self.name = name

        def on_start(self, *args, **kwargs):
            calls.append((self.name, "on_start", kwargs))

        def on_minibatch(self, *args, **kwargs):
            calls.append((self.name, "on_minibatch", kwargs))

        def on_epoch(self, *args, **kwargs):
            calls.append((self.name, "on_epoch", kwargs))

        def on_end(self, *args, **kwargs):
            calls.append((self.name, "on_end", kwargs))

    callbacks = CallbackList([_Recording("a"), _Recording("b")])

    callbacks.on_start(step=0, metrics={})
    callbacks.on_minibatch(step=1, metrics={})
    callbacks.on_epoch()
    callbacks.on_end(step=2, metrics={})

    assert [entry[0] for entry in calls] == ["a", "b"] * 4
    assert [entry[1] for entry in calls] == [
        "on_start",
        "on_start",
        "on_minibatch",
        "on_minibatch",
        "on_epoch",
        "on_epoch",
        "on_end",
        "on_end",
    ]
    assert calls[0][2] == {"step": 0, "metrics": {}}
    assert calls[-1][2] == {"step": 2, "metrics": {}}


def test_callback_list_tolerates_no_callbacks():
    callbacks = CallbackList()

    callbacks.on_start(step=0, metrics={})
    callbacks.on_rollout_end()
    callbacks.flush()


def test_callback_list_forwards_positional_arguments():
    seen: list[tuple] = []

    class _Recording(MinimalCollectorCallback):
        def on_env_step(self, *args, **kwargs):
            seen.append((args, kwargs))

    CallbackList([_Recording()]).on_env_step(3, info={}, extra=True)

    assert seen == [((3,), {"info": {}, "extra": True})]


# ---------------------------------------------------------------- metrics logger


def test_console_logger_formats_metrics(caplog):
    from rl_lib.tracking.console_logger import ConsoleMetricsLogger

    with caplog.at_level(logging.DEBUG, logger="rl_lib.tracking.console_logger"):
        ConsoleMetricsLogger().log_metrics({"loss/total": 1.5}, step=7)

    assert "Step 7" in caplog.text
    assert "'loss/total': '1.500000'" in caplog.text


def test_console_logger_formats_parameters(caplog):
    from rl_lib.tracking.console_logger import ConsoleMetricsLogger

    with caplog.at_level(logging.DEBUG, logger="rl_lib.tracking.console_logger"):
        ConsoleMetricsLogger().log_parameters({"lr": 1e-4})

    assert "'lr': '0.0001'" in caplog.text


def test_base_logger_artifact_logging_is_a_noop(tmp_path):
    class _Bare(MetricsLogger):
        def log_metrics(self, metrics, step): ...

    artifact = tmp_path / "video.mp4"
    artifact.write_bytes(b"data")

    _Bare().log_artifact(str(artifact))


# ---------------------------------------------------------------- params logger


def test_params_logger_logs_a_non_empty_config():
    logger = FakeLogger()
    callback = ParamsLoggingCallback(logger)

    callback.on_rollout_start(config={"epochs": 4, "minibatch_size": 256})

    assert logger.parameters == [{"epochs": 4, "minibatch_size": 256}]


@pytest.mark.parametrize("config", [None, {}])
def test_params_logger_ignores_an_empty_config(config):
    logger = FakeLogger()
    callback = ParamsLoggingCallback(logger)

    callback.on_rollout_start(config=config)
    callback.on_env_step(step=1)
    callback.on_rollout_end()
    callback.flush()

    assert logger.parameters == []
    assert logger.metrics == []


# ---------------------------------------------------------------- metrics logging


def test_minibatch_granularity_logs_every_minibatch():
    logger = FakeLogger()
    callback = MetricsLoggingCallback(logger, granularity="minibatch")

    callback.on_start(step=0, metrics={"rollout/returns": 1.0})
    callback.on_minibatch(metrics={"loss/total": 1.0}, step=10)
    callback.on_minibatch(metrics={"loss/total": 3.0}, step=10)
    callback.on_end(step=0, metrics={"training/lr_0": 1e-4})

    # entries that share a step are merged, so the rollout summary and the
    # training summary arrive as one record
    assert logger.metrics == [
        ({"rollout/returns": 1.0, "training/lr_0": 1e-4}, 0),
        ({"loss/total": 1.0}, 1),
        ({"loss/total": 3.0}, 2),
    ]


def test_epoch_granularity_averages_the_minibatches_of_an_epoch():
    logger = FakeLogger()
    callback = MetricsLoggingCallback(logger, granularity="epoch")

    callback.on_start(step=0, metrics={"rollout/returns": 1.0})
    callback.on_minibatch(metrics={"loss/total": 1.0, "loss/critic": 2.0})
    callback.on_minibatch(metrics={"loss/total": 3.0, "loss/critic": 4.0})
    callback.on_epoch()
    assert logger.metrics == [], "nothing is logged before the flush"

    callback.on_end(step=0, metrics=None)

    # the rollout summary keeps the trainer's step, the epoch average is
    # stamped with the callback's own minibatch counter
    assert logger.metrics == [
        ({"rollout/returns": 1.0}, 0),
        ({"loss/total": 2.0, "loss/critic": 3.0}, 1),
    ]


def test_epoch_granularity_does_not_average_empty_epochs():
    logger = FakeLogger()
    callback = MetricsLoggingCallback(logger, granularity="epoch")

    callback.on_start(step=0, metrics={"rollout/returns": 1.0})
    callback.on_epoch()
    callback.on_epoch()
    callback.on_end(step=0, metrics=None)

    assert logger.metrics == [({"rollout/returns": 1.0}, 0)]


def test_batch_granularity_defers_to_the_end_of_the_rollout():
    logger = FakeLogger()
    callback = MetricsLoggingCallback(logger, granularity="batch")

    callback.on_start(step=5, metrics={"rollout/returns": 1.0})
    callback.on_minibatch(metrics={"loss/total": 1.0})
    callback.on_minibatch(metrics={"loss/total": 5.0})
    callback.on_epoch()
    callback.on_end(step=5, metrics={"training/lr_0": 1e-4})

    assert logger.metrics == [
        (
            {
                "rollout/returns": 1.0,
                "training/lr_0": 1e-4,
                "loss/total": 3.0,
            },
            5,
        )
    ]


def test_flush_merges_same_step_entries_and_empties_the_buffer():
    logger = FakeLogger()
    callback = MetricsLoggingCallback(logger, granularity="minibatch")

    callback.on_start(step=1, metrics={"loss/total": 1.0})
    callback.on_start(step=1, metrics={"loss/critic": 2.0})
    callback.flush()

    assert logger.metrics == [({"loss/total": 1.0, "loss/critic": 2.0}, 1)]

    callback.flush()
    assert len(logger.metrics) == 1


def test_on_start_drops_minibatches_left_over_from_the_previous_rollout():
    logger = FakeLogger()
    callback = MetricsLoggingCallback(logger, granularity="epoch")

    callback.on_minibatch(metrics={"loss/total": 99.0})
    callback.on_start(step=1, metrics={"rollout/returns": 0.0})
    callback.on_minibatch(metrics={"loss/total": 1.0})
    callback.on_epoch()
    callback.on_end(step=1, metrics=None)

    logged = {key: value for metrics, _ in logger.metrics for key, value in metrics.items()}
    assert logged["loss/total"] == 1.0


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_callback_creates_the_folder(tmp_path, agent: Agent):
    path = tmp_path / "nested" / "checkpoints"

    CheckpointsSaveCallback(str(path), agent, intervals=2)

    assert path.is_dir()


def test_checkpoint_callback_saves_every_interval(tmp_path, agent: Agent):
    path = tmp_path / "checkpoints"
    callback = CheckpointsSaveCallback(str(path), agent, intervals=3)

    for _ in range(9):
        callback.on_end()

    saved = sorted(entry.name for entry in path.iterdir())
    assert saved == ["checkpoint_3.pt", "checkpoint_6.pt", "checkpoint_9.pt"]


def test_checkpoint_round_trips_the_agent(tmp_path, agent: Agent, observations):
    path = tmp_path / "checkpoints"
    callback = CheckpointsSaveCallback(str(path), agent, intervals=1)
    callback.on_end()

    obs = observations(2)
    agent.reset()
    T.manual_seed(123)
    before = agent.act(obs, T.zeros(2, dtype=T.bool))[0]

    restored = Agent(
        network=agent._network,
        observation_dim=1,
        action_dim=3,
        stack_size=agent.stack_size,
        device="cpu",
    )
    restored.load_state_dict(path / "checkpoint_1.pt")
    restored.reset()
    T.manual_seed(123)
    after = restored.act(obs, T.zeros(2, dtype=T.bool))[0]

    T.testing.assert_close(before, after)


def test_checkpoint_callback_ignores_the_other_hooks(tmp_path, agent: Agent):
    path = tmp_path / "checkpoints"
    callback = CheckpointsSaveCallback(str(path), agent, intervals=1)

    callback.on_start()
    callback.on_minibatch()
    callback.on_epoch()

    assert list(path.iterdir()) == []


# ---------------------------------------------------------------- statistics


def _episode_info(returns, lengths, done_mask=None, key: str = "episode"):
    done_mask = (
        np.ones(len(returns), dtype=bool)
        if done_mask is None
        else np.asarray(done_mask, dtype=bool)
    )
    return {
        "_episode": done_mask,
        key: {"r": np.asarray(returns, np.float64), "l": np.asarray(lengths, np.float64)},
    }


def test_statistics_step_mode_logs_per_step_averages():
    logger = FakeLogger()
    callback = RecordStatisticLoggerCallback(logger, mode="step")

    callback.on_env_step(step=4, info=_episode_info([2.0, 4.0], [10, 20]))

    assert logger.metrics == [
        ({"episode/returns": 3.0, "episode/lengths": 15.0, "episode/n": 2}, 4)
    ]


def test_statistics_step_mode_ignores_partial_episode_masks():
    logger = FakeLogger()
    callback = RecordStatisticLoggerCallback(logger, mode="step")

    callback.on_env_step(
        step=1,
        info=_episode_info([1.0, 5.0, 9.0], [2, 4, 6], done_mask=np.array([True, False, True])),
    )

    assert logger.metrics[0][0] == {"episode/returns": 5.0, "episode/lengths": 4.0, "episode/n": 2}


def test_statistics_step_mode_ignores_steps_without_episode_data():
    logger = FakeLogger()
    callback = RecordStatisticLoggerCallback(logger, mode="step")

    callback.on_env_step(step=0, info={})

    assert logger.metrics == []


def test_statistics_mean_mode_accumulates_until_the_rollout_ends():
    logger = FakeLogger()
    callback = RecordStatisticLoggerCallback(logger, mode="mean")

    callback.on_env_step(step=1, info=_episode_info([2.0, 4.0], [10, 20]))
    callback.on_env_step(step=2, info=_episode_info([6.0], [30]))
    assert logger.metrics == []

    callback.on_rollout_end()

    assert logger.metrics == [
        ({"episode/returns": 4.0, "episode/lengths": 20.0, "episode/n": 3}, 2)
    ]

    callback.flush()
    assert len(logger.metrics) == 1, "the counters are reset after the flush"


def test_statistics_mean_mode_flush_is_a_noop_without_episodes():
    logger = FakeLogger()
    callback = RecordStatisticLoggerCallback(logger, mode="mean")

    callback.on_rollout_end()
    callback.flush()

    assert logger.metrics == []


def test_statistics_step_mode_flush_is_a_noop():
    logger = FakeLogger()
    callback = RecordStatisticLoggerCallback(logger, mode="step")

    callback.on_env_step(step=1, info=_episode_info([2.0], [10]))
    callback.flush()

    assert len(logger.metrics) == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Gymnasium's SAME_STEP autoreset nests the terminal payload under "
        'info["final_info"], but the callback only looks at info["episode"]'
    ),
)
def test_statistics_reads_gymnasics_nested_final_info():
    logger = FakeLogger()
    callback = RecordStatisticLoggerCallback(logger, mode="step")

    callback.on_env_step(
        step=6,
        info={
            "_episode": np.array([True, False]),
            "final_info": [{"episode": {"r": np.array([3.0]), "l": np.array([12])}}, {}],
        },
    )

    assert logger.metrics == [
        ({"episode/returns": 3.0, "episode/lengths": 12.0, "episode/n": 1}, 6)
    ]


# ---------------------------------------------------------------- video


class _RecorderEnv(gym.Env):
    """Minimal env that also owns the two attributes `stop_video_recording` reads."""

    def __init__(self, name: str):
        self.video_folder = "videos"
        self._video_name = name
        self.stopped = 0

    def stop_recording(self):
        self.stopped += 1


def test_stop_video_recording_finds_the_wrapper_in_the_chain():
    recorder = _RecorderEnv("run")
    env = gym.Wrapper(gym.Wrapper(recorder))

    path = stop_video_recording(env)

    assert recorder.stopped == 1
    assert path == "videos/run.mp4"


def test_stop_video_recording_prefers_the_outermost_matching_wrapper():
    deepest = _RecorderEnv("deepest")
    shallowest = _RecorderEnv("shallowest")
    env = gym.Wrapper(gym.Wrapper(gym.Wrapper(deepest)))
    env = gym.Wrapper(env)
    env = gym.Wrapper(gym.Wrapper(shallowest))

    # the walk starts at the outermost wrapper, so that recorder wins
    assert stop_video_recording(env) == "videos/shallowest.mp4"
    assert (shallowest.stopped, deepest.stopped) == (1, 0)


def test_stop_video_recording_returns_none_without_a_recorder(caplog):
    import gymnasium as gym

    env = gym.Env()

    with caplog.at_level(logging.WARNING, logger="rl_lib.training.callbacks.utils"):
        assert stop_video_recording(env) is None

    assert "No RecordVideo wrapper found" in caplog.text


def test_stop_video_recording_falls_back_to_close_video_recorder():
    stopped: list[str] = []

    class _Recorder(gym.Env):
        video_folder = "videos"
        _video_name = "run"

        def close_video_recorder(self):
            stopped.append("close_video_recorder")

    assert stop_video_recording(_Recorder()) == "videos/run.mp4"
    assert stopped == ["close_video_recorder"]


@pytest.fixture
def video_callback(monkeypatch, agent: Agent):
    """A RecordVideoCallback whose env is a stub, so no game window is opened."""

    created: dict[str, Any] = {}

    def _make_env(env_id, num_envs, **kwargs):
        created.update({"env_id": env_id, "num_envs": num_envs, **kwargs})
        return _StubVideoEnv()

    monkeypatch.setattr("rl_lib.training.callbacks.record_video.make_env", _make_env)
    return RecordVideoCallback(
        env_id="CarRacing-v3",
        agent=agent,
        video_folder="videos",
        metrics_loggers=[],
    )


class _StubVideoEnv:
    num_envs = 1

    def __init__(self):
        self.envs = [types.SimpleNamespace()]
        self.closed = False

    def reset(self):
        return T.zeros(1, 96, 96, 1, dtype=T.uint8).numpy(), {}

    def step(self, action):
        observation = T.zeros(1, 96, 96, 1, dtype=T.uint8).numpy()
        return (
            observation,
            np.zeros(1, np.float32),
            np.ones(1, bool),
            np.zeros(1, bool),
            {"episode": {"r": np.array([12.0]), "l": np.array([40])}},
        )

    def close(self):
        self.closed = True


def test_video_callback_builds_its_own_single_env(video_callback: RecordVideoCallback):
    assert video_callback.env.num_envs == 1


def test_video_callback_records_on_the_interval(video_callback: RecordVideoCallback, monkeypatch):
    recorded: list[int] = []
    monkeypatch.setattr(RecordVideoCallback, "record", lambda self, step: recorded.append(step))

    video_callback.interval = 10
    for step in range(25):
        video_callback.on_env_step(step=step, info={})

    assert recorded == [0, 10, 20]


def test_video_callback_rollout_end_is_broken(monkeypatch, agent: Agent):
    """`on_rollout_end` calls `self.record()` although `record` requires a step."""
    monkeypatch.setattr(
        "rl_lib.training.callbacks.record_video.make_env", lambda *a, **k: _StubVideoEnv()
    )
    callback = RecordVideoCallback(
        env_id="CarRacing-v3",
        agent=agent,
        video_folder="videos",
        metrics_loggers=[],
    )

    with pytest.raises(TypeError):
        callback.on_rollout_end()

    assert callback.env.closed is False
