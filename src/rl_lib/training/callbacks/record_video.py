from logging import getLogger

from rl_lib.agent import Agent
from rl_lib.envs.make_env import make_env
from rl_lib.training.callbacks.base import CollectorCallback
from rl_lib.evaluation.inference import log_evaluation_results, run_inference
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.utils import stop_video_recording
from rl_lib.run_config import VideoCallbackSettings



logger = getLogger(__name__)


class RecordVideoCallback(CollectorCallback):
    def __init__(
        self,
        agent: Agent,
        metrics_loggers: list[MetricsLogger],
        config: VideoCallbackSettings,
    ):
        self.agent = agent
        self.cfg = config
        self._loggers = metrics_loggers
        self.env = make_env(config.environment)
        self._step = 0
        self._last_step = 0


    def record(self, step: int):
        logger.debug("Recording video...")
        results = run_inference(
            self.agent,
            self.env,
            episodes=1,
            seed=self.cfg.seed,
            temperature=self.agent.cfg.deterministic_temperature,
            episode_stats_key=self.cfg.stats_key,
        )

        video_path = stop_video_recording(self.env.envs[0])
        log_evaluation_results(results, self._loggers, step=step, scope="video")

        if video_path:
            [lgr.log_artifact(video_path, artifact_path=f"videos/step_{self._step}") for lgr in self._loggers]

        self._step += 1
        return

    def on_rollout_start(self, *args, **kwargs):
        pass

    def on_env_step(self, step: int, *args, **kwargs):
        self._last_step = step
        if step % self.cfg.interval == 0:
            self.record(step)


    def on_rollout_end(self, *args, **kwargs):
        # the collector ends a rollout without a step number, so continue from
        # the last one it reported; the environment is closed either way, and a
        # failed recording must not leave the policy in eval mode
        try:
            self.record(self._last_step + 1)
        finally:
            self.env.close()
            self.agent.train()

    def flush(self, *args, **kwargs):
        pass
