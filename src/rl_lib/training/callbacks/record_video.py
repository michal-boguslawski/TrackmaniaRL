"""Collector callback: records evaluation videos at regular intervals.

Runs a single evaluation episode with video recording enabled in the
environment, logs the video as an artifact, and logs episode metrics.
"""

from logging import getLogger

from rl_lib.agent import Agent
from rl_lib.envs.make_env import make_env
from rl_lib.training.callbacks.base import Callback
from rl_lib.evaluation.inference import log_evaluation_results, run_inference
from rl_lib.tracking.artifacts import step_artifact_path
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.utils import stop_video_recording
from rl_lib.run_config import VideoCallbackSettings


logger = getLogger(__name__)


class RecordVideoCallback(Callback):
    """Record policy evaluation videos at regular intervals.

    Creates a dedicated evaluation environment with video recording wrapper.
    At each interval (and at rollout end), runs one episode, saves the
    video file, and logs it as an artifact to all metrics loggers.

    Attributes:
        agent: Evaluation agent (copy of training agent).
        cfg: VideoCallbackSettings (interval, environment, seed, etc.).
        _loggers: MetricsLogger backends for artifact/metric logging.
        env: Video recording environment (single env, sync).
        _step: Counter for video artifact naming.
        _last_step: Last collector step seen (for final video timing).
    """

    def __init__(
        self,
        agent: Agent,
        metrics_loggers: list[MetricsLogger],
        config: VideoCallbackSettings,
    ):
        """Initialize video recording callback.

        Args:
            agent: Evaluation agent for recording.
            metrics_loggers: Tracking backends for video/metric logging.
            config: Video settings including environment, interval, seed.
        """
        self.agent = agent
        self.cfg = config
        self._loggers = metrics_loggers
        self.env = make_env(config.environment)
        self._step = 0
        self._last_step = 0

    def record(self, step: int):
        """Run one episode, capture video, log metrics and video artifact.

        Args:
            step: Training step for logging context.
        """
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
            artifact_path = step_artifact_path("videos", self._step)
            [lgr.log_artifact(video_path, artifact_path=artifact_path) for lgr in self._loggers]

        self._step += 1

    def on_rollout_start(self, *args, **kwargs):
        pass

    def on_env_step(self, step: int, *args, **kwargs):
        """Record video at configured step interval."""
        self._last_step = step
        if step % self.cfg.interval == 0:
            self.record(step)

    def on_rollout_end(self, *args, **kwargs):
        """Record final video, ensure agent returns to train mode."""
        # The collector ends a rollout without a step number, so continue from
        # the last one it reported; the environment is closed either way, and a
        # failed recording must not leave the policy in eval mode
        try:
            self.record(self._last_step + 1)
        finally:
            self.env.close()
            self.agent.train()

    def flush(self, *args, **kwargs):
        pass
