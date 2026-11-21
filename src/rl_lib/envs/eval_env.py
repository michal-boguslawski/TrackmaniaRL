from gymnasium.vector import AutoresetMode

from rl_lib.envs.settings.env import EnvironmentSettings
from rl_lib.envs.settings.wrappers.record_video import RecordVideoWrapperSettings


def eval_env_settings(
    base_env: EnvironmentSettings,
    *,
    recording: RecordVideoWrapperSettings | None = None,
) -> EnvironmentSettings:
    render_mode = "rgb_array" if recording else base_env.render_mode
    env_settings = base_env.model_copy(
        update={
            "num_envs": 1,
            "render_mode": render_mode,
            "vectorization_mode": "sync",
            "autoreset_mode": AutoresetMode.NEXT_STEP,
            "normalize_rewards": False,
            "wrappers": ([recording] if recording else []) + base_env.wrappers,
        }
    )
    return env_settings
