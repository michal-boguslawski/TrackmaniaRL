# training/ppo/builder.py
import torch as T

from rl_lib.checkpoint_loaders import load_pretrained
from rl_lib.networks.actor import Actor
from rl_lib.networks.critic import VCritic
from rl_lib.networks.encoder import Encoder
from rl_lib.policy.history import History
from rl_lib.policy.policy import Policy
from rl_lib.training.base import BuildContext
from rl_lib.training.ppo.modules import ActorCritic
from rl_lib.training.ppo.trainer import PPOTrainer


def build_ppo(ctx: BuildContext) -> tuple[Policy, PPOTrainer]:
    cfg, device = ctx.config, ctx.device

    # 1. construct
    encoder = Encoder(
        observation_dim=ctx.obs_shape[-1],
        stack_size=cfg.agent.stack_size,
        config=cfg.network.encoder,
        cnn_autocast_dtype=cfg.run.cnn_autocast,
    ).to(device)
    actor = Actor(
        action_dim=ctx.action_shape[-1],
        in_dim=encoder.out_dim,
        config=cfg.network.actor,
    ).to(device)
    critic = VCritic(encoder.out_dim, cfg.network.critic).to(device)
    modules = ActorCritic(
        encoder, actor, critic
    )  # nn.Module container, owns the parameters

    # 2. prepare weights, before anything copies or wraps them
    load_pretrained(modules, cfg.run, device)
    if device.type == "cuda" and cfg.run.channels_last:
        encoder.cnn.to(memory_format=T.channels_last)

    # 3. build consumers, both from the same objects
    policy = Policy(
        encoder,
        actor,
        History(encoder.stack_size),
    )
    trainer = PPOTrainer(
        modules, cfg.trainer, cfg.rollout, ctx
    )  # builds optimizer, buffer, scheduler
    return policy, trainer
