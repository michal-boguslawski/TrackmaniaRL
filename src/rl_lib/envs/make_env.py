from copy import copy
from gymnasium import make_vec, VectorizeMode
from gymnasium.vector import VectorEnv
from functools import partial

from src.rl_lib.envs.wrappers import WRAPPERS

def make_env(
    env_id: str,
    num_envs: int,
    skip: int = 4,
    video_folder: str = "./logs/videos",
    record: bool = False,
    continuous: bool = True,
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
    return env
