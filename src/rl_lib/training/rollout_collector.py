"""Rollout collector: orchestrates environment steps, buffer filling, and training updates.

Coordinates the interaction between vectorized environments, the agent policy,
the rollout buffer, and the PPO trainer. Manages callbacks for logging,
video recording, and checkpointing at the rollout level.
"""

import json

import numpy as np
import torch as T
from gymnasium.vector import VectorEnv
from tqdm import tqdm

from rl_lib.policy.policy import Policy
from rl_lib.run_config import RolloutSettings
from rl_lib.training.base import Trainer
from rl_lib.training.callbacks.base import Callback, CallbackList
from rl_lib.training.transitions import Transition


class RolloutCollector:
    """Collects rollouts from vectorized environments and triggers PPO updates.

    Runs a loop of environment steps, storing transitions in the rollout buffer.
    When the buffer is full, calls the trainer to perform PPO epochs over the
    collected data. Handles truncated episode bootstrap value computation
    using Gymnasium's final_observation from vector env info dict.

    Attributes:
        env: Vectorized Gymnasium environment.
        buffer: RolloutBuffer for storing transitions.
        # TODO
        cfg: RolloutSettings (buffer_size, epochs, minibatch_size, gamma, gae_lambda).
        seed: Environment seed.
    """

    def __init__(
        self,
        env: VectorEnv,
        policy: Policy,
        trainer: Trainer,
        callbacks: list[Callback] | None = None,
        seed: int | None = None,
    ):
        """Initialize the rollout collector.

        Args:
            env: Vectorized environment (must match buffer/trainer config).
            buffer: RolloutBuffer instance.
            agent # TODO
            config: RolloutSettings shared with buffer (validated for consistency).
            callbacks: Callbacks for rollout-level events.
            seed: Random seed for environment reset.

        Raises:
            ValueError: If buffer.cfg != config (epochs/minibatch_size or
                gamma/gae_lambda would disagree).
        """
        self.env = env
        self.policy = policy
        self.trainer = trainer
        self.seed = seed
        self._callbacks = CallbackList(callbacks)

    def _on_rollout_start(self, *args, **kwargs):
        """Notify callbacks that rollout collection is starting."""
        self._callbacks.on_rollout_start(*args, **kwargs)

    def _on_env_step(self, *args, **kwargs):
        """Notify callbacks of each environment step."""
        self._callbacks.on_env_step(*args, **kwargs)

    def _on_rollout_end(self, *args, **kwargs):
        """Notify callbacks that rollout collection has ended."""
        self._callbacks.on_rollout_end(*args, **kwargs)

    def _callback_flush(self, *args, **kwargs):
        """Flush callback buffers (e.g., write metrics to MLflow)."""
        self._callbacks.flush(*args, **kwargs)

    def run(
        self,
        training_steps: int,
        profiler: T.profiler.profile | None = None,
    ) -> None:
        """Main collection loop.

        Args:
            training_steps: Total number of vector-environment steps to collect.
            profiler: Optional active PyTorch profiler. Its schedule advances
                once per vector-environment iteration.
        """
        state, _ = self.env.reset(seed=self.seed)
        done = T.zeros(self.env.num_envs, dtype=T.bool).to(self.env.device)
        # self._on_rollout_start(config=self.config(training_steps))

        for i in tqdm(range(training_steps)):
            action, extras = self.policy.act(state, done)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            transition = Transition.from_step(
                state, action, extras, next_state, reward, terminated, truncated, info
            )
            self.trainer.observe(transition)

            if "final_obs" in info:
                final_observations = info.get(
                    "final_obs", info.get("final_observation")
                )
                if final_observations is None:
                    raise RuntimeError(
                        "Vector environment truncated an episode without providing its final observation"
                    )

                final_mask = info.get("_final_obs", info.get("_final_observation"))
                truncated_indices = (
                    transition.truncated.nonzero(as_tuple=True)[0].cpu().numpy()
                )
                if final_mask is not None:
                    mask = np.asarray(final_mask, dtype=np.bool_)
                    if not mask[truncated_indices].all():
                        raise RuntimeError(
                            "Vector environment did not provide a final observation for every truncated episode"
                        )

            state = transition.next_obs
            self._on_env_step(step=i, info=info)

            self.trainer.maybe_update(i)

            self._callback_flush()

            if profiler is not None:
                profiler.step()

        self._on_rollout_end()
