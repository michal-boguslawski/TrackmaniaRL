from contextlib import closing
from pathlib import Path

import torch as T

from rl_lib.agent import Agent
from rl_lib.envs.make_env import make_env
from rl_lib.networks.factory import Network
from rl_lib.run_config import load_config


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    episodes: int = 5,
    record_video: bool = False,
) -> list[float]:
    """Run a deterministic policy for a fixed number of episodes."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    config = load_config(str(config_path))
    device = T.device(
        "cuda" if config.run.device == "auto" and T.cuda.is_available()
        else "cpu" if config.run.device == "auto"
        else config.run.device
    )
    video_folder = str(Path(config.callbacks.video_folder) / "evaluation")
    environment_config = config.environment.model_copy(update={
        "num_envs": 1,
        "vectorization_mode": "sync",
        "normalize_rewards": False,
        "record_video": record_video,
        "video_folder": video_folder,
    })

    with closing(make_env(environment_config)) as env:
        network = Network(
            observation_dim=env.observation_space.shape[-1],
            action_dim=env.action_space.shape[-1],
            stack_size=config.agent.stack_size,
            config=config.network,
        ).to(device)
        agent = Agent(network, device, config.agent)
        agent.load_state_dict(str(checkpoint_path))
        agent.eval()

        observation, _ = env.reset(seed=config.run.seed)
        done = T.zeros(1, dtype=T.bool, device=device)
        episode_return = 0.0
        episode_length = 0
        returns = []

        while len(returns) < episodes:
            observation_tensor = T.from_numpy(observation).to(device)
            action, _, _ = agent.act(
                observation_tensor,
                done,
                temperature=config.agent.deterministic_temperature,
            )
            observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            episode_return += float(reward[0])
            episode_length += 1
            done = T.as_tensor(terminated | truncated, dtype=T.bool, device=device)

            if bool(done[0]):
                returns.append(episode_return)
                print(
                    f"Episode {len(returns)}/{episodes}: "
                    f"return={episode_return:.2f}, steps={episode_length}"
                )
                episode_return = 0.0
                episode_length = 0

    if record_video:
        print(f"Evaluation videos saved to {video_folder}")
    return returns
