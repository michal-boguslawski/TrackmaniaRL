"""Vectorized environment factory with configurable wrappers.

Builds Gymnasium vector environments from EnvironmentSettings, applying
wrappers in order and optionally adding reward normalization.
"""

from copy import copy
from gymnasium import make_vec, VectorizeMode
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.wrappers.vector import NormalizeReward
from functools import partial

from rl_lib.envs.wrappers.registry import WRAPPERS
from rl_lib.envs.wrappers.settings import make_wrapper_settings
from rl_lib.run_config import EnvironmentSettings


def make_env(config: EnvironmentSettings) -> VectorEnv:
    """Create a vectorized environment with configured wrappers.

    Args:
        config: EnvironmentSettings with id, num_envs, wrappers, etc.

    Returns:
        VectorEnv with all wrappers applied and optional reward normalization.

    Wrapper order:
    1. record_video (if config.record_video or wrapper present)
    2. All wrappers from config.wrappers in order
    3. NormalizeReward (if config.normalize_rewards)

    Video recording forces SYNC vectorization mode.
    """
    _wrappers = copy(config.wrappers)
    configured_video = any(wrapper.name == "record_video" for wrapper in _wrappers)
    if config.record_video and not configured_video:
        _wrappers.insert(
            0,
            make_wrapper_settings("record_video", video_folder=config.video_folder),
        )
    record_enabled = config.record_video or configured_video

    wrappers_fn = []
    for wrapper in _wrappers:
        settings = make_wrapper_settings(wrapper) if isinstance(wrapper, str) else wrapper
        options = settings.model_dump(exclude={"name"})
        for optional_key in ("skip", "stack_size", "video_folder"):
            if options.get(optional_key) is None:
                options.pop(optional_key, None)
        options.setdefault("video_folder", config.video_folder)
        options.setdefault("skip", config.skip)
        if settings.name == "frame_stack" and "stack_size" not in options:
            raise ValueError("frame_stack wrapper requires stack_size")
        wrappers_fn.append(partial(WRAPPERS[settings.name], **options))

    env = make_vec(
        config.id,
        num_envs=config.num_envs,
        vectorization_mode=VectorizeMode.SYNC if record_enabled else VectorizeMode(config.vectorization_mode),
        # SAME_STEP keeps terminal actions as transitions and avoids emitting
        # a separate reset-only step on the next call to env.step().
        vector_kwargs={"autoreset_mode": AutoresetMode[config.autoreset_mode.upper()]},
        render_mode="rgb_array" if record_enabled else None,
        continuous=config.continuous,
        wrappers=wrappers_fn,
    )

    if config.normalize_rewards:
        env = NormalizeReward(
            env,
            gamma=config.reward_normalization_gamma,
            epsilon=config.reward_normalization_epsilon,
        )
    return env
