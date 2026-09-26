from copy import copy
from gymnasium import make_vec, VectorizeMode
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.wrappers.vector import NormalizeReward
from functools import partial

from rl_lib.envs.wrappers.registry import WRAPPERS
from rl_lib.run_config import EnvironmentSettings, WrapperSettings

def make_env(
    env_id: str,
    num_envs: int,
    skip: int = 4,
    video_folder: str = "./logs/videos",
    record: bool = False,
    continuous: bool = True,
    normalize_rewards: bool = False,
    vectorization_mode: str = "async",
    wrappers: list[str | WrapperSettings | dict] | None = None,
    autoreset_mode: str = "same_step",
    reward_normalization_gamma: float | None = None,
    reward_normalization_epsilon: float | None = None,
) -> VectorEnv:
    _wrappers = copy(wrappers or [])
    configured_video = any(
        (wrapper.name if isinstance(wrapper, WrapperSettings) else wrapper.get("name") if isinstance(wrapper, dict) else wrapper)
        == "record_video"
        for wrapper in _wrappers
    )
    if record and not configured_video:
        _wrappers.insert(0, WrapperSettings(name="record_video", video_folder=video_folder))
    record_enabled = record or configured_video

    wrappers_fn = []
    for wrapper in _wrappers:
        if isinstance(wrapper, str):
            settings = WrapperSettings(name=wrapper)
        elif isinstance(wrapper, WrapperSettings):
            settings = wrapper
        else:
            settings = WrapperSettings.model_validate(wrapper)
        if settings.name not in WRAPPERS:
            raise ValueError(f"Unknown environment wrapper: {settings.name}")
        options = settings.model_dump(exclude={"name"})
        for optional_key in ("skip", "stack_size", "video_folder"):
            if options[optional_key] is None:
                options.pop(optional_key)
        options.setdefault("video_folder", video_folder)
        options.setdefault("skip", skip)
        if settings.name == "frame_stack" and "stack_size" not in options:
            raise ValueError("frame_stack wrapper requires stack_size")
        wrappers_fn.append(partial(WRAPPERS[settings.name], **options))
    env = make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode=VectorizeMode.SYNC if record_enabled else VectorizeMode(vectorization_mode),
        # SAME_STEP keeps terminal actions as transitions and avoids emitting
        # a separate reset-only step on the next call to env.step().
        vector_kwargs={"autoreset_mode": AutoresetMode[autoreset_mode.upper()]},
        render_mode="rgb_array" if record_enabled else None,
        continuous=continuous,
        wrappers=wrappers_fn,
    )

    if normalize_rewards:
        environment_defaults = EnvironmentSettings()
        env = NormalizeReward(
            env,
            gamma=environment_defaults.reward_normalization_gamma if reward_normalization_gamma is None else reward_normalization_gamma,
            epsilon=environment_defaults.reward_normalization_epsilon if reward_normalization_epsilon is None else reward_normalization_epsilon,
        )
    return env
