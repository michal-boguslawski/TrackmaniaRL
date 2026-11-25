"""Epoch and minibatch iteration for PPO updates."""

from collections.abc import Callable
from logging import getLogger

import numpy as np
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
    train_step: Callable[..., dict[str, float]],
    metrics_logger: Callback | None = None,
    kl_control_settings: KLControlSettings | None = None,
    # current_update_step: Callable[[], int],
    # on_minibatch: Callable[..., None],
    # on_epoch: Callable[..., None],
) -> None:
    """Run PPO minibatch updates and apply configured KL early stopping.

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
        on_epoch: Callback dispatcher invoked after each epoch, including a
            partially completed epoch stopped by the KL threshold.
    """
    for epoch in range(optimization_settings.epochs):
        epoch_kls = []
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
            epoch_kls.append(minibatch_metrics["ppo/approx_kl"].item())

            if (
                kl_control_settings is not None
                and minibatch_metrics["ppo/approx_kl"]
                > kl_control_settings.target_kl * kl_control_settings.stop_multiplier
                and training_step > kl_control_settings.warmup_steps
                and kl_control_settings.hard_stop
            ):
                logger.warning(
                    "Hard stop mid-epoch: KL %.4f",
                    minibatch_metrics["ppo/approx_kl"],
                )
                break

        mean_epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else None
        metrics_logger.on_event(
            CallbackEvent.EPOCH_END,
            step=training_step,
        )
        if (
            kl_control_settings is not None
            and mean_epoch_kl is not None
            and mean_epoch_kl > kl_control_settings.target_kl
            and training_step > kl_control_settings.warmup_steps
            and kl_control_settings.hard_stop
        ):
            logger.warning(
                "Early stop epoch %s: KL %.4f > %.4f",
                epoch,
                mean_epoch_kl,
                kl_control_settings.target_kl,
            )
            break
