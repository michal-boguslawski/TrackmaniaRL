"""Collector callback: records evaluation videos at regular intervals.

Runs a single evaluation episode with video recording enabled in the
environment, logs the video as an artifact, and logs episode metrics.
"""

from logging import getLogger

import torch as T

from rl_lib.envs.eval_env import eval_env_settings
from rl_lib.envs.make_env import make_env
from rl_lib.envs.settings.env import EnvironmentSettings
from rl_lib.envs.settings.wrappers.record_video import RecordVideoWrapperSettings
from rl_lib.evaluation.inference import run_inference
from rl_lib.evaluation.utils import prepare_mean_stats_from_results
from rl_lib.paths import RunPaths
from rl_lib.policy.policy import Policy
from rl_lib.tracking.artifacts import step_artifact_path
from rl_lib.tracking.callbacks.base import Callback, CallbackEvent
from rl_lib.tracking.callbacks.utils import stop_video_recording
from rl_lib.tracking.config.callbacks.record_video import RecordVideoCallbackSettings
from rl_lib.tracking.sinks.base import MetricsSink

logger = getLogger(__name__)


class RecordVideoCallback(Callback):
    """Record policy evaluation videos at regular intervals.

    Creates a dedicated evaluation environment with video recording wrapper.
    At each interval (and at rollout end), runs one episode, saves the
    video file, and logs it as an artifact to all metrics loggers.

    Attributes:
        agent: Evaluation agent (copy of training agent).
        cfg: RecordVideoCallbackSettings (interval, folder, etc.).
        _loggers: MetricsLogger backends for artifact/metric logging.
        env: Video recording environment (single env, sync).
        _step: Counter for video artifact naming.
        _last_step: Last collector step seen (for final video timing).
    """

    def __init__(
        self,
        logger: MetricsSink,
        config: RecordVideoCallbackSettings,
        base_env: EnvironmentSettings,
        run_paths: RunPaths,
        device: T.device | str = "cpu",
        run_seed: int | None = None,
        *args,
        **kwargs,
    ):
        """Initialize video recording callback.

        Args:
            agent: Evaluation agent for recording.
            metrics_loggers: Tracking backends for video/metric logging.
            config: Video settings including environment, interval, seed.
        """
        self.cfg = config
        self._logger = logger
        self.device = device
        self.run_seed = run_seed

        self._env = make_env(
            eval_env_settings(
                base_env,
                recording=RecordVideoWrapperSettings(
                    video_folder=str(run_paths.video_dir),
                    episode_trigger=lambda _: True,
                ),
            ),
            device=device,
        )

    def record(self, policy: Policy) -> dict[str, float]:
        """Run one episode, capture video, log metrics and video artifact.

        Args:
            step: Training step for logging context.
        """
        logger.debug("Recording video...")
        results = run_inference(
            policy.fork(),
            self._env,
            episodes=1,
            seed=self.run_seed,
            device=self.device,
        )

        video_path = stop_video_recording(self._env.env.envs[0])

        logger.debug(f"Video saved in {video_path}")

        # log_evaluation_results(results, self._loggers, step=step, scope="video")

        # if video_path:
        #     artifact_path = step_artifact_path("videos", self._step)
        #     [
        #         lgr.log_artifact(video_path, artifact_path=artifact_path)
        #         for lgr in self._loggers
        #     ]
        return prepare_mean_stats_from_results(results, "video/episode")

    def on_event(
        self, event: CallbackEvent, step: int, policy: Policy | None = None, **kwargs
    ) -> None:
        if (
            event is CallbackEvent.ENV_STEP and step % self.cfg.interval == 0
        ) or event is CallbackEvent.ROLLOUT_END:
            if policy is None:
                raise TypeError("Missing policy")
            metrics = self.record(policy)
            self._logger.log_metrics(metrics, step)
        if event is CallbackEvent.ROLLOUT_END:
            self._env.close()

    def flush(self) -> None: ...
