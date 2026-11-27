"""CPU regression test for gradient flow through each PPO module group."""

import torch as T

from rl_lib.buffers.settings.rollout import RolloutSettings
from rl_lib.networks.actor import Actor
from rl_lib.networks.critic import VCritic
from rl_lib.networks.encoder import Encoder
from rl_lib.networks.settings.actor import ActorSettings
from rl_lib.networks.settings.cnn import CNNSettings
from rl_lib.networks.settings.critic import VCriticSettings
from rl_lib.networks.settings.encoder import EncoderConfig
from rl_lib.networks.settings.layer import ConvLayerSettings, LinearLayerSettings
from rl_lib.networks.settings.temporal import TemporalSettings
from rl_lib.policy.masks import episode_mask
from rl_lib.training.ppo.config.learning_rates import PPOLearningRateSettings
from rl_lib.training.ppo.config.loss import PPOLossConfig
from rl_lib.training.ppo.config.optimization import PPOOptimizationSettings
from rl_lib.training.ppo.config.trainer import PPOTrainerSettings
from rl_lib.training.ppo.metrics.build import build_ppo_metrics_factory
from rl_lib.training.ppo.metrics.contexts import PPOUpdateContext
from rl_lib.training.ppo.modules import ActorCritic
from rl_lib.training.ppo.trainer import PPOTrainer
from rl_lib.training.settings.loss import EntropySettings


def _make_trainer(learning_rate: float = 1e-4) -> tuple[PPOTrainer, ActorCritic]:
    """Build a compact PPO trainer and network for CPU minibatch tests.

    Args:
        learning_rate: Learning rate applied to every optimizer parameter group.

    Returns:
        The trainer and its actor-critic modules.
    """
    encoder = Encoder(
        observation_dim=1,
        config=EncoderConfig(
            cnn=CNNSettings(
                conv_layers=[ConvLayerSettings(out_channels=4, kernel_size=3)],
                adaptive_pool_size=2,
                hidden_layers=[LinearLayerSettings(out_dim=8)],
                out_dim=8,
            ),
            temporal=TemporalSettings(out_dim=8),
            stack_size=2,
        ),
    )
    actor = Actor(
        action_dim=3,
        in_dim=encoder.out_dim,
        settings=ActorSettings(hidden_layers=[LinearLayerSettings(out_dim=8)]),
        action_low=[-1.0, 0.0, 0.0],
        action_high=[1.0, 1.0, 1.0],
    )
    critic = VCritic(encoder.out_dim, VCriticSettings())
    modules = ActorCritic(encoder, actor, critic).to("cpu")
    settings = PPOTrainerSettings(
        loss=PPOLossConfig(entropy=EntropySettings(decay=1.0)),
        optimizer={
            "learning_rates": PPOLearningRateSettings(
                cnn_lr=learning_rate,
                sequence_encoder_lr=learning_rate,
                actor_lr=learning_rate,
                critic_lr=learning_rate,
            )
        },
        optimization=PPOOptimizationSettings(minibatch_size=4, epochs=1),
        rollout=RolloutSettings(steps_per_env=4),
    )
    trainer = PPOTrainer(
        modules,
        settings,
        total_steps=4,
        metrics_factory=build_ppo_metrics_factory(),
    )
    return trainer, modules


def _make_minibatch(modules: ActorCritic) -> dict[str, T.Tensor]:
    """Create one fixed minibatch from the initial policy."""
    batch_size, sequence_length = 4, modules.encoder.stack_size
    observations = T.randint(
        0,
        256,
        (batch_size, sequence_length, 16, 16, 1),
        dtype=T.uint8,
    )
    dones = T.zeros(batch_size, sequence_length, dtype=T.bool)

    with T.no_grad():
        initial_dist, initial_values, _ = modules(observations, episode_mask(dones))
        actions = initial_dist.sample()
        old_log_probs = initial_dist.log_prob(actions)

    return {
        "observation": observations,
        "action": actions,
        "old_log_probs": old_log_probs,
        "old_values": initial_values,
        "returns": initial_values + T.tensor([1.0, -0.5, 0.75, -1.25]),
        "advantages": T.tensor([1.0, -0.7, 0.3, 1.2]),
        "dones": dones,
    }


def _minibatch_objective(
    trainer: PPOTrainer, minibatch: dict[str, T.Tensor]
) -> T.Tensor:
    """Recompute the PPO objective for an unchanged minibatch."""
    with T.no_grad():
        action_dist, values, action_mean = trainer.modules(
            minibatch["observation"], episode_mask(minibatch["dones"])
        )
        update_context = PPOUpdateContext(
            values=values,
            returns=minibatch["returns"],
            dist=action_dist,
            old_log_probs=minibatch["old_log_probs"],
            new_log_probs=action_dist.log_prob(minibatch["action"]),
            advantages=minibatch["advantages"],
            actions=action_mean,
            old_values=minibatch["old_values"],
            actor_loss_cfg=trainer.cfg.loss.actor_loss_cfg,
            entropy_decay=trainer.losses.entropy_decay,
        )
        return trainer.losses.calculate_losses(update_context).total


def test_representative_minibatch_reaches_every_parameter_group() -> None:
    """A PPO update should produce nonzero gradients in all four module groups."""
    trainer, modules = _make_trainer()
    batch_size, sequence_length = 4, modules.encoder.stack_size
    observations = T.randint(
        0,
        256,
        (batch_size, sequence_length, 16, 16, 1),
        dtype=T.uint8,
    )
    dones = T.zeros(batch_size, sequence_length, dtype=T.bool)

    with T.no_grad():
        initial_dist, initial_values, _ = modules(observations, episode_mask(dones))
        actions = initial_dist.sample()
        old_log_probs = initial_dist.log_prob(actions)

    trainer.train_step(
        observation=observations,
        action=actions,
        old_log_probs=old_log_probs,
        old_values=initial_values,
        returns=initial_values + T.tensor([1.0, -0.5, 0.75, -1.25]),
        advantages=T.tensor([1.0, -0.7, 0.3, 1.2]),
        dones=dones,
    )

    parameter_groups = {
        "cnn": modules.encoder.cnn.parameters(),
        "temporal": modules.encoder.sequence_encoder.parameters(),
        "actor": modules.actor.parameters(),
        "critic": modules.critic.parameters(),
    }
    for group_name, parameters in parameter_groups.items():
        gradients = [parameter.grad for parameter in parameters]
        assert gradients and all(gradient is not None for gradient in gradients), (
            f"{group_name} did not receive gradients for every parameter"
        )
        group_norm = T.linalg.vector_norm(
            T.stack([T.linalg.vector_norm(gradient) for gradient in gradients])
        )
        assert T.isfinite(group_norm) and group_norm.item() > 0.0, (
            f"{group_name} gradient norm was {group_norm.item()}"
        )


def test_one_optimizer_step_decreases_the_same_minibatch_objective() -> None:
    """A modest optimizer step should not increase its fixed minibatch objective."""
    trainer, modules = _make_trainer(learning_rate=1e-3)
    minibatch = _make_minibatch(modules)
    objective_before = _minibatch_objective(trainer, minibatch)

    trainer.train_step(**minibatch)

    objective_after = _minibatch_objective(trainer, minibatch)
    tolerance = 1e-5
    assert T.isfinite(objective_after)
    assert objective_after <= objective_before + tolerance, (
        f"objective increased from {objective_before.item():.6f} "
        f"to {objective_after.item():.6f}"
    )


def test_excessive_learning_rate_overshoots_the_minibatch_objective() -> None:
    """A deliberately excessive rate should fail the descent tolerance."""
    trainer, modules = _make_trainer(learning_rate=10.0)
    minibatch = _make_minibatch(modules)
    objective_before = _minibatch_objective(trainer, minibatch)

    trainer.train_step(**minibatch)

    objective_after = _minibatch_objective(trainer, minibatch)
    tolerance = 1e-5
    assert T.isfinite(objective_after)
    assert objective_after > objective_before + tolerance, (
        f"expected overshoot beyond {tolerance:g}, got "
        f"{objective_before.item():.6f} -> {objective_after.item():.6f}"
    )
