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
from rl_lib.training.ppo.config.optimization import PPOOptimizationSettings
from rl_lib.training.ppo.config.trainer import PPOTrainerSettings
from rl_lib.training.ppo.metrics.build import build_ppo_metrics_factory
from rl_lib.training.ppo.modules import ActorCritic
from rl_lib.training.ppo.trainer import PPOTrainer


def _make_trainer() -> tuple[PPOTrainer, ActorCritic]:
    """Build a compact PPO trainer and network for a CPU minibatch test."""
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
        optimizer={
            "learning_rates": PPOLearningRateSettings(
                cnn_lr=1e-4,
                sequence_encoder_lr=1e-4,
                actor_lr=1e-4,
                critic_lr=1e-4,
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
