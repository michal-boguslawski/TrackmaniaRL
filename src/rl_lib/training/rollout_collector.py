from gymnasium.vector import VectorEnv
import numpy as np
import torch as T
from tqdm import tqdm

from src.rl_lib.buffers.rollout_buffer import RolloutBuffer, RolloutStep
from src.rl_lib.training.ppo_trainer import PPOTrainer
from src.rl_lib.training.callbacks.base import CollectorCallback, CallbackList


class RolloutCollector:
    def __init__(
        self,
        env: VectorEnv,
        buffer: RolloutBuffer,
        trainer: PPOTrainer,
        epochs: int,
        minibatch_size: int,
        callbacks: list[CollectorCallback] | None = None,
    ):
        self.env = env
        self.buffer = buffer
        self.trainer = trainer
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self._callbacks = CallbackList(callbacks)

    def _on_rollout_start(self, *args, **kwargs):
        self._callbacks.on_rollout_start(*args, **kwargs)

    def _on_env_step(self, *args, **kwargs):
        self._callbacks.on_env_step(*args, **kwargs)

    def _on_rollout_end(self, *args, **kwargs):
        self._callbacks.on_rollout_end(*args, **kwargs)

    def run(self, training_steps: int):
        state, _ = self.env.reset()
        self.buffer.reset()
        self._on_rollout_start()

        for i in tqdm(range(training_steps)):
            state = T.from_numpy(state).to(self.trainer.device)
            action, log_probs, critic_value = self.trainer.act(state)
            next_state, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())

            self.buffer.add(RolloutStep(
                observation=state,
                action=action,
                critic_value=critic_value,
                old_log_probs=log_probs,
                reward=T.from_numpy(reward),
                terminated=T.from_numpy(terminated),
                truncated=T.from_numpy(truncated),
            ))

            state = next_state
            self._on_env_step(step=i, info=info)

            if self.buffer.is_full():
                self.trainer.train(self.buffer.get(), epochs=self.epochs, minibatch_size=self.minibatch_size)
                self.buffer.reset()

        self._on_rollout_end()
