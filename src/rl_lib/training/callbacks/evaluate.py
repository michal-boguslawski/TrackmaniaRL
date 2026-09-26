from __future__ import annotations

from rl_lib.agent import Agent
from rl_lib.envs.make_env import make_env
from rl_lib.evaluation.inference import log_evaluation_results, run_inference
from rl_lib.evaluation.runtime import _episode_stats_key
from rl_lib.run_config import EvaluationCallbackSettings
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.base import CollectorCallback


class EvaluationCallback(CollectorCallback):
    """Evaluate the current policy periodically and after training completes."""

    def __init__(
        self,
        agent: Agent,
        metrics_loggers: list[MetricsLogger],
        config: EvaluationCallbackSettings,
    ):
        self.agent = Agent(agent.network, agent.device, agent.cfg)
        self.cfg = config
        self._loggers = metrics_loggers
        self.env = make_env(config.environment)
        self._training_steps: int | None = None
        self._evaluation_index = 0
        self._last_evaluation_step: int | None = None

    def _evaluate(self, episodes: int, step: int, scope: str) -> None:
        seed = (self.cfg.seed + self._evaluation_index) % (2**32)
        self._evaluation_index += 1
        results = run_inference(
            self.agent,
            self.env,
            episodes=episodes,
            seed=seed,
            temperature=self.agent.cfg.deterministic_temperature,
            episode_stats_key=_episode_stats_key(self.cfg.environment),
        )
        log_evaluation_results(results, self._loggers, step=step, scope=scope)
        self._last_evaluation_step = step

    def on_rollout_start(self, config: dict | None = None, *args, **kwargs) -> None:
        if config:
            self._training_steps = config.get("training_steps")

    def on_env_step(self, step: int, *args, **kwargs) -> None:
        completed_steps = step + 1
        if completed_steps % self.cfg.interval != 0:
            return
        # The final evaluation supersedes a periodic one on the last step.
        if self._training_steps is not None and completed_steps >= self._training_steps:
            return
        self._evaluate(self.cfg.episodes, completed_steps, "periodic")

    def on_rollout_end(self, *args, **kwargs) -> None:
        final_step = self._training_steps or (self._last_evaluation_step or 0)
        try:
            self._evaluate(self.cfg.final_episodes, final_step, "final")
        finally:
            self.env.close()

    def flush(self, *args, **kwargs) -> None:
        pass
