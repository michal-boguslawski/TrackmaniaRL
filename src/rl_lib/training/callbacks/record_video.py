from logging import getLogger
import torch as T

from rl_lib.agent import Agent
from rl_lib.envs.make_env import make_env
from rl_lib.training.callbacks.base import CollectorCallback
from rl_lib.tracking.base import MetricsLogger
from rl_lib.training.callbacks.utils import stop_video_recording


logger = getLogger(__name__)


class RecordVideoCallback(CollectorCallback):
    def __init__(
        self,
        env_id: str,
        agent: Agent,
        video_folder: str,
        metrics_loggers: list[MetricsLogger],
        skip: int | None = None,
        wrappers: list[str] | None = None,
        interval: int = 10_000
    ):
        self.agent = agent
        self.video_folder = video_folder
        self._loggers = metrics_loggers
        self.env = make_env(
            env_id,
            1,
            skip=skip,
            video_folder=video_folder,
            record=True,
            vectorization_mode="sync",
            wrappers=wrappers
        )
        self.interval = interval
        self._step = 0

    def record(self, step: int):
        self.agent.eval()
        self.agent.reset()
        logger.debug("Recording video...")
        state, _ = self.env.reset()
        done = T.zeros(self.env.num_envs, dtype=T.bool).to(self.agent.device)
        while not done.any():
            state, _, _, _, _, _, _, _, done, info = self.agent.step_env(self.env, state, done, temperature=1e-4)

        video_path = stop_video_recording(self.env.envs[0])
        
        metrics = {
            "evaluation/episode_returns": float(info["episode"]["r"][0]),
            "evaluation/episode_lengths": float(info["episode"]["l"][0]),
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
