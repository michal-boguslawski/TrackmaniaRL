from copy import copy
from gymnasium import make_vec, VectorizeMode
from gymnasium.vector import VectorEnv
from gymnasium.wrappers.vector import NormalizeReward
from functools import partial

from rl_lib.envs.wrappers.registry import WRAPPERS

def make_env(
    env_id: str,
    num_envs: int,
    skip: int = 4,
    video_folder: str = "./logs/videos",
    record: bool = False,
    continuous: bool = True,
    normalize_rewards: bool = False,
    vectorization_mode: str = "async",
    wrappers: list[str] | None = None
) -> VectorEnv:
    _wrappers = copy(wrappers or [])
    if record:
        _wrappers.insert(0, "record_video")

    wrappers_fn = [
        partial(
            WRAPPERS[wrapper_name],
            video_folder=video_folder,
            skip=skip,
        )
        for wrapper_name in _wrappers
    ]
    env = make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode=VectorizeMode.SYNC if record else VectorizeMode(vectorization_mode),
        render_mode="rgb_array" if record else None,
        continuous=continuous,
        wrappers=wrappers_fn,
    )

    if normalize_rewards:
        env = NormalizeReward(env)
    return env
