from rl_lib.run_config import RunConfig
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.evaluate import EvaluationCallback
from rl_lib.training.callbacks.factory import create_callbacks
from rl_lib.training.callbacks.metrics_logger import MetricsLoggingCallback
from rl_lib.training.callbacks.model_checkpoint import ModelCheckpointCallback
from rl_lib.training.callbacks.params_logger import ParamsLoggingCallback
from rl_lib.training.callbacks.record_statistics import RecordStatisticLoggerCallback


class _Logger(MetricsLogger):
    def log_metrics(self, metrics, step):
        pass


class _UnusedEnv:
    def close(self):
        pass


def test_callback_factory_creates_configured_callbacks_in_their_groups(
    monkeypatch, agent, network_config
):
    config = RunConfig.model_validate({
        "environment": {
            "id": "unused",
            "wrappers": [{"name": "record_episode_stats"}],
        },
        "agent": {"stack_size": agent.stack_size},
        "network": network_config.model_dump(),
        "callbacks": [
            {"name": "checkpoints", "interval": 200},
            {"name": "metrics"},
            {"name": "episode_statistics"},
            {"name": "evaluation", "interval": 10, "episodes": 2, "final_episodes": 3},
        ],
    })
    monkeypatch.setattr(
        "rl_lib.training.callbacks.evaluate.make_env", lambda environment: _UnusedEnv()
    )

    groups = create_callbacks(
        config,
        agent,
        agent,
        [_Logger()],
        run_config={"run": "test"},
    )

    assert [type(callback) for callback in groups.trainer] == [MetricsLoggingCallback]
    assert [type(callback) for callback in groups.collector] == [
        ParamsLoggingCallback,
        ModelCheckpointCallback,
        RecordStatisticLoggerCallback,
        EvaluationCallback,
    ]
    assert groups.collector[1]._interval == 200
