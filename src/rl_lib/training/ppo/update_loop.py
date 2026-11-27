"""Epoch and minibatch iteration for PPO updates."""

from collections.abc import Callable
from logging import getLogger

import torch as T

from rl_lib.tracking.callbacks.base import Callback, CallbackEvent
from rl_lib.training.ppo.config.optimization import PPOOptimizationSettings
from rl_lib.training.ppo.config.trainer import KLControlSettings
from rl_lib.training.ppo.minibatches import get_iid_minibatches

logger = getLogger(__name__)


def run_ppo_update_loop(
    batch: dict[str, T.Tensor],
    stack_size: int,
    optimization_settings: PPOOptimizationSettings,
    training_step: int,
    train_step: Callable[..., dict[str, T.Tensor]],
    metrics_logger: Callback | None = None,
    kl_control_settings: KLControlSettings | None = None,
    # current_update_step: Callable[[], int],
    # on_minibatch: Callable[..., None],
    # on_epoch: Callable[..., None],
) -> None:
    """Run PPO epochs and check the KL stopping condition at epoch boundaries.

    Args:
        batch: Rollout batch with temporal observation and done windows.
        epochs: Number of passes over the rollout.
        minibatch_size: Maximum transitions per minibatch.
        stack_size: Temporal window size used by the agent.
        cfg: Current trainer settings controlling minibatch mode and KL stops.
        training_step: Global environment step, used for warmup and logging.
        train_step: Callable performing one minibatch optimizer update.
        current_update_step: Callable returning the trainer's current optimizer
            update index for callback logging.
        on_minibatch: Callback dispatcher receiving ``metrics`` and ``step``.
        on_epoch: Callback dispatcher invoked after each complete epoch.
    """
    for epoch in range(optimization_settings.epochs):
        epoch_kls: list[T.Tensor] = []
        minibatches = get_iid_minibatches(
            batch,
            optimization_settings.minibatch_size,
            stack_size,
            shuffle=optimization_settings.minibatch_indexing_mode == "iid",
        )
        for minibatch in minibatches:
            try:
                minibatch_metrics = train_step(**minibatch)
            except Exception as error:
                logger.error(
                    "Error at training step %s, epoch %s: %s",
                    training_step,
                    epoch,
                    error,
                )
                raise
            if (minibatch_metrics is not None) and (metrics_logger is not None):
                metrics_logger.on_event(
                    CallbackEvent.MINIBATCH_END,
                    step=training_step,
                    metrics=minibatch_metrics,
                )
            should_check_kl = (
                kl_control_settings is not None
                and kl_control_settings.hard_stop
                and training_step > kl_control_settings.warmup_steps
            )
            if should_check_kl:
                assert kl_control_settings is not None
                approx_kl_value = minibatch_metrics["update/approx_kl"]
                assert isinstance(approx_kl_value, T.Tensor)
                assert not approx_kl_value.requires_grad
                epoch_kls.append(approx_kl_value)

        mean_epoch_kl = T.stack(epoch_kls).mean() if epoch_kls else None
        metrics_logger.on_event(
            CallbackEvent.EPOCH_END,
            step=training_step,
        )
        if (
            kl_control_settings is not None
            and mean_epoch_kl is not None
            and training_step > kl_control_settings.warmup_steps
            and kl_control_settings.hard_stop
        ):
            assert isinstance(mean_epoch_kl, T.Tensor)
            if bool((mean_epoch_kl > kl_control_settings.target_kl).item()):
                logger.warning(
                    "Early stop after epoch %s: mean approximate KL exceeded %.4f",
                    epoch,
                    kl_control_settings.target_kl,
                )
                break
