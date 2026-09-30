"""Dump every metric name PPO training emits, grouped by prefix.

`docs/metrics.md` documents what each metric means and what to cross-check it
against. That table drifts silently: a metric added to the losses never has to
touch the docs, and nothing fails when the two disagree. Regenerate the name
inventory with this script and fold the result back in:

    uv run --extra cpu python scripts/benchmarks/metric_inventory.py

It collects one short rollout through a tiny CPU network and runs a full update
at ``Verbosity.ALL``, so the emitted names are the real ones rather than the
ones a reader assumes. Emitting a name here still proves nothing about whether
its value is meaningful; only the tests do that.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch as T

from rl_lib.agent import Agent
from rl_lib.buffers.rollout_buffer import RolloutBuffer, RolloutStep
from rl_lib.networks.config import (
    ActorConfig,
    CNNConfig,
    CriticConfig,
    ConvLayerConfig,
    LinearLayerConfig,
    NetworkConfig,
    TemporalConfig,
)
from rl_lib.networks.factory import Network
from rl_lib.run_config import AgentSettings, RolloutSettings, TrainerSettings
from rl_lib.training.callbacks.base import Callback
from rl_lib.training.ppo.trainer import PPOTrainer

NUM_ENVS = 2
SIZE = 4
STACK_SIZE = 2
ACTION_DIM = 3
OBSERVATION_SHAPE = (96, 96, 1)


class _MetricCapture(Callback):
    """Collect the metric names the trainer hands to its callbacks.

    ``MetricsLoggingCallback`` buffers the same values before passing them to a
    backend, so intercepting here sees the same set without needing one.
    """

    def __init__(self) -> None:
        self.names: list[str] = []

    def _collect(self, metrics: dict[str, float] | None) -> None:
        if metrics:
            self.names.extend(metrics)

    def on_start(self, *args, metrics=None, **kwargs) -> None:
        self._collect(metrics)

    def on_minibatch(self, *args, metrics=None, **kwargs) -> None:
        self._collect(metrics)

    def on_epoch(self, *args, metrics=None, **kwargs) -> None:
        self._collect(metrics)

    def on_end(self, *args, metrics=None, **kwargs) -> None:
        self._collect(metrics)


def _tiny_network_config() -> NetworkConfig:
    """Production topology with tiny widths, so the dump stays fast on CPU."""
    return NetworkConfig(
        cnn=CNNConfig(
            conv_layers=[
                ConvLayerConfig(out_channels=2, kernel_size=8, stride=4),
                ConvLayerConfig(out_channels=4, kernel_size=4, stride=2),
                ConvLayerConfig(out_channels=8, kernel_size=3),
                ConvLayerConfig(out_channels=8, kernel_size=3, stride=2),
            ],
            hidden_layers=[LinearLayerConfig(out_dim=8)],
            out_dim=8,
        ),
        temporal=TemporalConfig(out_dim=8),
        actor=ActorConfig(hidden_layers=[LinearLayerConfig(out_dim=8, activation="gelu")]),
        critic=CriticConfig(hidden_layers=[LinearLayerConfig(out_dim=8, activation="gelu")]),
    )


def _collect_rollout(agent: Agent) -> dict[str, T.Tensor]:
    """Drive a few steps through the agent so stored log-probs and values are the
    ones the update path has to reproduce."""
    buffer = RolloutBuffer(RolloutSettings(buffer_size=SIZE), stack_size=STACK_SIZE)
    done = T.zeros(NUM_ENVS, dtype=T.bool)
    for _ in range(SIZE):
        state = T.randint(0, 256, (NUM_ENVS, *OBSERVATION_SHAPE), dtype=T.uint8)
        action, log_probs, value = agent.act(state, done)
        buffer.add(
            RolloutStep(
                observation=state,
                action=action,
                critic_value=value,
                old_log_probs=log_probs,
                reward=T.rand(NUM_ENVS),
                terminated=T.zeros(NUM_ENVS, dtype=T.bool),
                truncated=T.zeros(NUM_ENVS, dtype=T.bool),
                truncated_value=T.zeros(NUM_ENVS),
            )
        )
    return buffer.get()


def emitted_metric_names() -> list[str]:
    """Return every metric name a full PPO update sends to a backend."""
    T.manual_seed(0)
    network = Network(
        observation_dim=OBSERVATION_SHAPE[-1],
        action_dim=ACTION_DIM,
        stack_size=STACK_SIZE,
        config=_tiny_network_config(),
    )
    agent = Agent(network=network, device="cpu", config=AgentSettings(stack_size=STACK_SIZE))
    capture = _MetricCapture()
    trainer = PPOTrainer(agent, TrainerSettings(), callbacks=[capture], verbosity=2)
    trainer.train(_collect_rollout(agent), epochs=1, minibatch_size=2, training_step=0)
    return sorted(set(capture.names))


def group_by_prefix(names: list[str]) -> dict[str, list[str]]:
    """Group metric names by their ``prefix/`` namespace."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        grouped[name.split("/", 1)[0]].append(name)
    return dict(sorted(grouped.items()))


def main() -> int:
    """Print the inventory as a Markdown table body."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        nargs="*",
        help="only show these prefixes (default: all)",
    )
    args = parser.parse_args()

    grouped = group_by_prefix(emitted_metric_names())
    for prefix, names in grouped.items():
        if args.namespace and prefix not in args.namespace:
            continue
        print(f"### `{prefix}/`\n")
        for name in names:
            print(f"- `{name}`")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())