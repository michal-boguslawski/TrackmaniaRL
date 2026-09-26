from copy import copy
from gymnasium import make_vec, VectorizeMode
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.wrappers.vector import NormalizeReward
from functools import partial

from rl_lib.envs.wrappers.registry import WRAPPERS
from rl_lib.run_config import EnvironmentSettings, WrapperSettings

def make_env(config: EnvironmentSettings) -> VectorEnv:
    _wrappers = copy(config.wrappers)
    configured_video = any(wrapper.name == "record_video" for wrapper in _wrappers)
    if config.record_video and not configured_video:
        _wrappers.insert(0, WrapperSettings(name="record_video", video_folder=config.video_folder))
    record_enabled = config.record_video or configured_video

    wrappers_fn = []
    for wrapper in _wrappers:
        settings = wrapper if isinstance(wrapper, WrapperSettings) else WrapperSettings(name=wrapper)
        if settings.name not in WRAPPERS:
            raise ValueError(f"Unknown environment wrapper: {settings.name}")
        options = settings.model_dump(exclude={"name"})
        for optional_key in ("skip", "stack_size", "video_folder"):
            if options[optional_key] is None:
                options.pop(optional_key)
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

