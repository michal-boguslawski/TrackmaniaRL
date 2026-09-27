"""Epoch and minibatch iteration for PPO updates."""

from collections.abc import Callable
from logging import getLogger

import numpy as np
import torch as T

from rl_lib.run_config import TrainerSettings
from rl_lib.training.ppo.minibatches import get_iid_minibatches


logger = getLogger(__name__)


def run_ppo_update_loop(
    batch: dict[str, T.Tensor],
    epochs: int,
    minibatch_size: int,
    stack_size: int,
    cfg: TrainerSettings,
    training_step: int,
    train_step: Callable[..., dict[str, float]],
    current_update_step: Callable[[], int],
    on_minibatch: Callable[..., None],
    on_epoch: Callable[..., None],
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
    for epoch in range(epochs):
        epoch_kls = []
        minibatches = get_iid_minibatches(
            batch,
            minibatch_size,
            stack_size,
            shuffle=cfg.minibatch_indexing_mode == "iid",
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
            if not minibatch_metrics:
                continue
            epoch_kls.append(minibatch_metrics["metrics/approx_kl"])

            if (
                minibatch_metrics["metrics/approx_kl"]
                > cfg.target_kl * cfg.kl_stop_multiplier
                and training_step > cfg.kl_warmup_steps
                and cfg.hard_stop_kl
            ):
                logger.warning(
                    "Hard stop mid-epoch: KL %.4f",
                    minibatch_metrics["metrics/approx_kl"],
                )
                break

            on_minibatch(
                metrics=minibatch_metrics, step=current_update_step()
            )

        mean_epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else None
        on_epoch()
        if (
            mean_epoch_kl is not None
            and mean_epoch_kl > cfg.target_kl
            and training_step > cfg.kl_warmup_steps
            and cfg.hard_stop_kl
        ):
            logger.warning(
                "Early stop epoch %s: KL %.4f > %.4f",
                epoch,
                mean_epoch_kl,
                cfg.target_kl,
            )
            break
