"""Rollout collector: orchestrates environment steps, buffer filling, and training updates.

Coordinates the interaction between vectorized environments, the agent policy,
the rollout buffer, and the PPO trainer. Manages callbacks for logging,
video recording, and checkpointing at the rollout level.
"""

from gymnasium.vector import VectorEnv
import json
import numpy as np
import torch as T
from tqdm import tqdm

from rl_lib.buffers.rollout_buffer import RolloutBuffer, RolloutStep
from rl_lib.run_config import RolloutSettings
from rl_lib.training.ppo_trainer import PPOTrainer
from rl_lib.training.callbacks.base import CollectorCallback, CallbackList


class RolloutCollector:
    """Collects rollouts from vectorized environments and triggers PPO updates.

    Runs a loop of environment steps, storing transitions in the rollout buffer.
    When the buffer is full, calls the trainer to perform PPO epochs over the
    collected data. Handles truncated episode bootstrap value computation
    using Gymnasium's final_observation from vector env info dict.

    Attributes:
        env: Vectorized Gymnasium environment.
        buffer: RolloutBuffer for storing transitions.
        trainer: PPOTrainer for policy updates.
        cfg: RolloutSettings (buffer_size, epochs, minibatch_size, gamma, gae_lambda).
        seed: Environment seed.
        run_config: Full run config dict for MLflow logging.
    """

    def __init__(
        self,
        env: VectorEnv,
        buffer: RolloutBuffer,
        trainer: PPOTrainer,
        config: RolloutSettings,
        callbacks: list[CollectorCallback] | None = None,
        seed: int | None = None,
        run_config: dict | None = None,
    ):
        """Initialize the rollout collector.

        Args:
            env: Vectorized environment (must match buffer/trainer config).
            buffer: RolloutBuffer instance.
            trainer: PPOTrainer instance.
            config: RolloutSettings shared with buffer (validated for consistency).
            callbacks: List of CollectorCallbacks for rollout-level events.
            seed: Random seed for environment reset.
            run_config: Full config dict for MLflow parameter logging.

        Raises:
            ValueError: If buffer.cfg != config (epochs/minibatch_size or
                gamma/gae_lambda would disagree).
        """
        if buffer.cfg != config:
            raise ValueError(
                "the collector and its buffer must share one RolloutSettings; "
                "epochs/minibatch_size and gamma/gae_lambda would otherwise disagree"
            )
        self.env = env
        self.buffer = buffer
        self.trainer = trainer
        self.cfg = config
        self.seed = seed
        self.run_config = run_config or {}
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

    def run(self, training_steps: int):
        """Main collection loop.

        Args:
            training_steps: Total number of vector-environment steps to collect.
        """
        state, _ = self.env.reset(seed=self.seed)
        done = T.zeros(self.env.num_envs, dtype=T.bool).to(self.trainer.device)
        self.buffer.reset()
        self._on_rollout_start(config=self.config(training_steps))

        for i in tqdm(range(training_steps)):
            (next_state, state, action, log_probs, critic_value, reward, terminated, truncated, done, info) = self.trainer.step_env(self.env, state, done)

            truncated_value = T.zeros_like(
                truncated,
                dtype=T.float32,
                device=self.trainer.device,
            )
            if truncated.any():
                final_observations = info.get("final_obs", info.get("final_observation"))
                if final_observations is None:
                    raise RuntimeError(
                        "Vector environment truncated an episode without providing its final observation"
                    )

                final_mask = info.get("_final_obs", info.get("_final_observation"))
                truncated_indices = truncated.nonzero(as_tuple=True)[0].cpu().numpy()
                if final_mask is not None:
                    mask = np.asarray(final_mask, dtype=np.bool_)
                    if not mask[truncated_indices].all():
                        raise RuntimeError(
                            "Vector environment did not provide a final observation for every truncated episode"
                        )

                bootstrap_observations = next_state.copy()
                for env_index in truncated_indices:
                    bootstrap_observations[env_index] = final_observations[env_index]
                bootstrap_tensor = T.from_numpy(bootstrap_observations).to(self.trainer.device)
                values = self.trainer.bootstrap_value(bootstrap_tensor)
                truncated_value[truncated] = values[truncated]

            self.buffer.add(RolloutStep(
                observation=state,
                action=action,
                critic_value=critic_value,
                old_log_probs=log_probs,
                reward=T.from_numpy(reward).to(T.float32).to(self.trainer.device),
                terminated=terminated,
                truncated=truncated,
                truncated_value=truncated_value,
            ))

            state = next_state
            self._on_env_step(step=i, info=info)

            if self.buffer.is_full():
                self.trainer.train(
                    self.buffer.get(),
                    epochs=self.cfg.epochs,
                    minibatch_size=self.cfg.minibatch_size,
                    training_step=i,
                )

                self.buffer.reset_counter()
                self._callback_flush()

        self._on_rollout_end()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"epochs={self.cfg.epochs}, "
            f"minibatch_size={self.cfg.minibatch_size}, "
            f"num_envs={self.env.num_envs}, "
            f"buffer={self.buffer!r}, "
            f"trainer={self.trainer!r})"
        )

    def config(self, training_steps: int | None = None) -> dict[str, int | float | str]:
        """Flat, MLflow-loggable config for the whole training run, merging
        this collector's own params with buffer/trainer/network configs.
        """
        merged: dict[str, int | float | str] = {
            "epochs": self.cfg.epochs,
            "minibatch_size": self.cfg.minibatch_size,
            "num_envs": self.env.num_envs,
            "gamma": self.buffer.gamma,
            "gae_lambda": self.buffer.gae_lambda,
        }

        if training_steps is not None:
            merged["training_steps"] = training_steps
        def flatten(prefix: str, value):
            if isinstance(value, dict):
                for key, nested in value.items():
                    flatten(f"{prefix}.{key}" if prefix else key, nested)
            elif isinstance(value, list):
                merged[prefix] = json.dumps(value, sort_keys=True)
            else:
                merged[prefix] = value
        flatten("", self.run_config)
        merged.update({f"buffer.{k}": v for k, v in self.buffer.config().items()})
        merged.update({f"trainer.{k}": v for k, v in self.trainer.config().items()})
        merged.update({f"network.{k}": v for k, v in self.trainer.network_config().items()})
        return merged
