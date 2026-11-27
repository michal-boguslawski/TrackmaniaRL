import torch as T

from rl_lib.training.ppo.metrics.contexts import PPOUpdateContext


def calculate_approx_kl(
    update_context: PPOUpdateContext,
    include_per_action: bool = True,
) -> dict[str, T.Tensor]:
    """Compute approximate KL, optionally including per-action diagnostics.

    Args:
        update_context: PPO update tensors, with old and new log probabilities
            shaped ``(batch, action_dim)``.
        include_per_action: Whether to reduce and return one KL metric per
            action dimension.

    Returns:
        Metric names mapped to zero-dimensional tensors. ``approx_kl`` is
        always returned for PPO training control.
    """
    assert isinstance(update_context, PPOUpdateContext)
    log_ratios = update_context.new_log_probs - update_context.old_log_probs
    approx_kls = log_ratios.exp() - 1 - log_ratios
    metrics = {"approx_kl": approx_kls.sum(-1).mean()}
    if include_per_action:
        metrics.update(
            {
                f"approx_kl_{i}": mean_value
                for i, mean_value in enumerate(approx_kls.mean(0))
            }
        )
    return metrics


def calculate_clip_fraction(
    update_context: PPOUpdateContext,
) -> dict[str, T.Tensor]:
    assert isinstance(update_context, PPOUpdateContext)
    log_ratios = update_context.new_log_probs - update_context.old_log_probs

    ratios = log_ratios.exp()
    epsilon = update_context.actor_loss_cfg.ppo_epsilon

    joint_ratio = ratios.prod(dim=-1)

    metrics = {
        "clip_fraction": ((joint_ratio - 1).abs() > epsilon).float().mean(),
    }

    action_clip_fractions = ((ratios - 1).abs() > epsilon).float().mean(dim=0)

    for i, fraction in enumerate(action_clip_fractions):
        metrics[f"action_clip_fraction/{i}"] = fraction

    return metrics


def calculate_entropy_coef(
    update_context: PPOUpdateContext,
) -> dict[str, T.Tensor]:
    assert isinstance(update_context, PPOUpdateContext)
    return {
        "entropy_coef": update_context.new_log_probs.new_tensor(
            update_context.entropy_decay.coefficient
        )
    }
