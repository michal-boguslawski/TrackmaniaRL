"""Collector callback: periodic and final policy evaluation.

Creates a copy of the training agent (same network, different instance)
and runs evaluation episodes in a separate environment at configured
intervals and at the end of training.
"""

from __future__ import annotations

from logging import getLogger

import torch as T

from rl_lib.envs.eval_env import eval_env_settings
from rl_lib.envs.make_env import make_env
from rl_lib.envs.settings.env import EnvironmentSettings
from rl_lib.evaluation.inference import prepare_stats_from_results, run_inference
from rl_lib.policy.policy import Policy
from rl_lib.tracking.callbacks.base import Callback, CallbackEvent
from rl_lib.tracking.config.callbacks.evaluation import EvaluationCallbackSettings
from rl_lib.tracking.sinks.base import MetricsSink

logger = getLogger(__name__)


class EvaluationCallback(Callback):
    """Evaluate the current policy periodically and after training completes.

    Uses a copy of the training agent (shares network weights) to run
    deterministic episodes in a fresh environment. Logs results with
    'periodic' or 'final' scope to distinguish scheduled vs final eval.

    Attributes:
        agent: Copy of training agent for evaluation.
        cfg: EvaluationCallbackSettings (interval, episodes, etc.).
        _loggers: MetricsLogger backends.
        env: Evaluation environment (single env, sync, no reward normalization).
        _training_steps: Total steps from rollout config (for final eval timing).
        _evaluation_index: Counter for seeding periodic evaluations.
        _last_evaluation_step: Step of last periodic evaluation.
    """

    def __init__(
        self,
        logger: MetricsSink,
        config: EvaluationCallbackSettings,
        base_env: EnvironmentSettings,
        device: T.device | str = "cpu",
        run_seed: int | None = None,
        *args,
        **kwargs,
    ):
        """Initialize evaluation callback.

        Args:
            agent: Training agent (network weights shared, new instance created).
            metrics_loggers: Tracking backends for result logging.
            config: Evaluation settings including environment, interval, episodes.
        """
        self.cfg = config
        self._logger = logger
        self.device = device
        self.run_seed = run_seed

        self._env = make_env(
            eval_env_settings(
                base_env,
                num_envs=self.cfg.num_envs,
            ),
            device=device,
        )

    def _evaluate(self, policy: Policy, episodes: int) -> dict[str, float]:
        """Run evaluation episodes and log results."""

        logger.debug(f"Start evaluation on {self.cfg.on_event}")
        results = run_inference(
            policy.fork(),
            self._env,
            episodes=episodes,
            seed=self.run_seed,
            device=self.device,
        )
        return prepare_stats_from_results(results, "evaluation")

    def on_event(
        self, event: CallbackEvent, step: int, policy: Policy | None = None, **kwargs
    ) -> None:
        if event is self.cfg.on_event and step % self.cfg.interval == 0:
            metrics = self._evaluate(policy, self.cfg.episodes)
            self._logger.log_metrics(metrics, step)

    def flush(self, *args, **kwargs) -> None:
        pass
