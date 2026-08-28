import numpy as np
import torch as T

from src.rl_lib.logger_setup import setup_logging, shutdown_logging
from src.rl_lib.envs.make_env import make_env
from src.rl_lib.training.rollout_collector import RolloutCollector
from src.rl_lib.buffers.rollout_buffer import RolloutBuffer
from src.rl_lib.networks.factory import Network
from src.rl_lib.training.ppo_trainer import PPOTrainer
from src.rl_lib.tracking.console_logger import ConsoleMetricsLogger
from src.rl_lib.training.callbacks.metrics_logger import MetricsLoggingCallback
from src.rl_lib.training.callbacks.record_statistics import RecordStatisticLoggerCallback
from src.rl_lib.training.callbacks.record_video import RecordVideoCallback
from src.rl_lib.training.callbacks.checkpoints_save import CheckpointsSaveCallback
from src.rl_lib.agent import Agent


BATCH_SIZE = 1024
NUM_ENVS = 2
STACK_SIZE = 4
SKIP = 2
MINIBATCH_SIZE = 256
EPOCHS = 4
DEVICE = T.device("cuda" if T.cuda.is_available() else "cpu")

setup_logging()

def main():
    env = make_env(
        "CarRacing-v3",
        num_envs=NUM_ENVS,
        # skip=SKIP,
        record=False,
        stack_size=STACK_SIZE,
        wrappers=[
            "record_episode_stats",
            "grayscale",
            # "max_and_skip",
        ]
    )
    
    buffer = RolloutBuffer(
        size=BATCH_SIZE,
        stack_size=STACK_SIZE,
        # num_envs=NUM_ENVS,
        # observation_space=env.observation_space.shape[-3:],
        # action_space=env.action_space.shape[-1:]
    )
    
    network = Network(
        env.observation_space.shape[-1],
        env.action_space.shape[-1],
        STACK_SIZE
    ).to(DEVICE)
    
    agent = Agent(
        network=network,
        observation_dim=env.observation_space.shape[-1],
        action_dim=env.action_space.shape[-1],
        stack_size=STACK_SIZE,
        device=DEVICE
    )
    record_agent = Agent(
        network=network,
        observation_dim=env.observation_space.shape[-1],
        action_dim=env.action_space.shape[-1],
        stack_size=STACK_SIZE,
        device=DEVICE
    )

    console_metrics_logger = ConsoleMetricsLogger()
    trainer = PPOTrainer(
        agent=agent,
        entropy_coef=5e-3,
        # entropy_decay=0.95,
        advantage_normalization_strategy="global",
        callbacks=[
            CheckpointsSaveCallback("./logs/checkpoints", agent, intervals=200),
            MetricsLoggingCallback(console_metrics_logger, granularity="batch"),
        ]
    )

    rng = np.random.default_rng(seed=42)

    training_steps = 1_000_000
    rollout_collector = RolloutCollector(
        env,
        buffer,
        trainer,
        EPOCHS,
        MINIBATCH_SIZE,
        callbacks=[
            RecordStatisticLoggerCallback(console_metrics_logger),
            RecordVideoCallback(
                "CarRacing-v3",
                agent=record_agent,
                video_folder="./logs/videos",
                metrics_logger=console_metrics_logger,
                skip=SKIP,
                wrappers=[
                    "record_episode_stats",
                    "grayscale",
                    # "max_and_skip",
                ],
                interval=50_000,
            ),
        ]
    )

    rollout_collector.run(training_steps)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_logging()
