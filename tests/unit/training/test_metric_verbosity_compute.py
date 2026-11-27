"""Tests that verbosity disables diagnostic metric producers."""

import pytest
import torch as T
from pydantic import ValidationError

from rl_lib.tracking.callbacks.base import Callback, CallbackEvent
from rl_lib.tracking.verbosity import Verbosity
from rl_lib.training.ppo.config.optimization import PPOOptimizationSettings
from rl_lib.training.ppo.config.trainer import KLControlSettings
from rl_lib.training.ppo.metrics.build import build_ppo_metrics_factory
from rl_lib.training.ppo.metrics.contexts import PPOUpdateContext
from rl_lib.training.ppo.metrics.update.policy import calculate_approx_kl
from rl_lib.training.ppo.update_loop import run_ppo_update_loop


def test_core_verbosity_does_not_register_optional_metric_calculations() -> None:
    for verbosity in (Verbosity.NONE, Verbosity.CORE):
        factory = build_ppo_metrics_factory(verbosity)

        assert factory.rollout.calculate(object()) == {}
        assert factory.optimization.calculate(object()) == {}
        assert factory.update.calculate(object()) == {}


def test_approx_kl_skips_per_action_reductions_when_not_requested() -> None:
    context = PPOUpdateContext(
        values=T.zeros(2),
        returns=T.zeros(2),
        dist=None,
        old_log_probs=T.zeros((2, 3)),
        new_log_probs=T.ones((2, 3)),
        advantages=T.zeros(2),
        actions=T.zeros((2, 3)),
        old_values=T.zeros(2),
        actor_loss_cfg=None,
        entropy_decay=None,
    )

    metrics = calculate_approx_kl(context, include_per_action=False)
    detailed_metrics = calculate_approx_kl(context)

    assert set(metrics) == {"approx_kl"}
    assert metrics["approx_kl"].ndim == 0
    assert set(detailed_metrics) == {
        "approx_kl",
        "approx_kl_0",
        "approx_kl_1",
        "approx_kl_2",
    }


def test_all_verbosity_registers_rollout_optimization_and_diagnostic_metrics() -> None:
    factory = build_ppo_metrics_factory(Verbosity.ALL)

    assert len(factory.rollout._functions) == 3
    assert len(factory.optimization._functions) == 2
    assert len(factory.update._functions) == 5


def test_kl_settings_no_longer_accept_a_stop_multiplier() -> None:
    with pytest.raises(ValidationError):
        KLControlSettings(stop_multiplier=3.0)


def test_kl_stopping_checks_after_the_epoch_and_forwards_tensor_metrics() -> None:
    class _CaptureCallback(Callback):
        def __init__(self) -> None:
            self.events: list[tuple[CallbackEvent, dict | None]] = []

        def on_event(self, event: CallbackEvent, **kwargs) -> None:
            self.events.append((event, kwargs.get("metrics")))

        def flush(self) -> None:
            pass

    batch = {
        "action": T.zeros((1, 4, 1)),
        "observation": T.zeros((1, 4, 1, 1, 1), dtype=T.uint8),
        "old_log_probs": T.zeros((1, 4, 1)),
        "critic_value": T.zeros((1, 4)),
        "returns": T.zeros((1, 4)),
        "advantages": T.zeros((1, 4)),
        "dones": T.zeros((1, 4), dtype=T.bool),
    }
    callback = _CaptureCallback()
    train_calls = 0

    def train_step(**kwargs) -> dict[str, T.Tensor]:
        nonlocal train_calls
        train_calls += 1
        return {"update/approx_kl": T.tensor(1.0)}

    run_ppo_update_loop(
        batch=batch,
        stack_size=1,
        optimization_settings=PPOOptimizationSettings(
            minibatch_size=2,
            epochs=3,
            minibatch_indexing_mode="sequential",
        ),
        training_step=1,
        train_step=train_step,
        metrics_logger=callback,
        kl_control_settings=KLControlSettings(
            target_kl=0.1,
            warmup_steps=0,
            hard_stop=True,
        ),
    )

    minibatch_metrics = [
        metrics
        for event, metrics in callback.events
        if event is CallbackEvent.MINIBATCH_END
    ]
    epoch_events = [
        event for event, _ in callback.events if event is CallbackEvent.EPOCH_END
    ]
    assert train_calls == 2
    assert len(minibatch_metrics) == 2
    assert all(T.is_tensor(metrics["update/approx_kl"]) for metrics in minibatch_metrics)
    assert epoch_events == [CallbackEvent.EPOCH_END]
