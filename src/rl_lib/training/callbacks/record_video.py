from logging import getLogger
import torch as T

from rl_lib.agent import Agent
from rl_lib.envs.make_env import make_env
from rl_lib.training.callbacks.base import CollectorCallback
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.utils import stop_video_recording
from rl_lib.run_config import CallbackSettings, WrapperSettings


logger = getLogger(__name__)


class RecordVideoCallback(CollectorCallback):
    def __init__(
        self,
        env_id: str,
        agent: Agent,
        video_folder: str,
        metrics_loggers: list[MetricsLogger],
        skip: int | None = None,
        wrappers: list[str | WrapperSettings] | None = None,
        interval: int = 10_000,
        seed: int | None = None,
        autoreset_mode: str = "same_step",
        normalize_rewards: bool = False,
        reward_normalization_gamma: float | None = None,
        reward_normalization_epsilon: float | None = None,
        video_recording: WrapperSettings | None = None,
        stats_key: str | None = None,
    ):
        self.agent = agent
        self.video_folder = video_folder
        self._loggers = metrics_loggers
        self.seed = seed
        self.autoreset_mode = autoreset_mode
        self.stats_key = stats_key or CallbackSettings().episode_statistics_key
        video_wrappers = [video_recording] if video_recording is not None else []
        video_wrappers.extend(wrappers or [])
        self.env = make_env(
            env_id,
            1,
            skip=skip,
            video_folder=video_folder,
            record=video_recording is None,
            normalize_rewards=normalize_rewards,
            vectorization_mode="sync",
            wrappers=video_wrappers,
            autoreset_mode=autoreset_mode,
            reward_normalization_gamma=reward_normalization_gamma,
            reward_normalization_epsilon=reward_normalization_epsilon,
        )
        self.interval = interval
        self._step = 0

    def record(self, step: int):
        self.agent.eval()
        self.agent.reset()
        logger.debug("Recording video...")
        state, _ = self.env.reset(seed=self.seed)
        done = T.zeros(self.env.num_envs, dtype=T.bool).to(self.agent.device)
        while not done.any():
            state, _, _, _, _, _, _, _, done, info = self.agent.step_env(
                self.env, state, done, temperature=self.agent.cfg.deterministic_temperature
            )

        video_path = stop_video_recording(self.env.envs[0])
        
        metrics = {
            "evaluation/episode_returns": float(info[self.stats_key]["r"][0]),
            "evaluation/episode_lengths": float(info[self.stats_key]["l"][0]),
        }
        [lgr.log_metrics(metrics, step=step) for lgr in self._loggers]

        if video_path:
            [lgr.log_artifact(video_path, artifact_path=f"videos/step_{self._step}") for lgr in self._loggers]

        self.agent.train()
        self._step += 1
        return

    def on_rollout_start(self, *args, **kwargs):
        pass

    def on_env_step(self, step: int, *args, **kwargs):
        if step % self.interval == 0:
            self.record(step)

    def on_rollout_end(self, *args, **kwargs):
        self.record()
        self.env.close()

    def flush(self, *args, **kwargs):
        pass
