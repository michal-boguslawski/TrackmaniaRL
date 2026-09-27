"""Utility for stopping video recording in Gymnasium wrapper chains."""

import gymnasium as gym
from logging import getLogger


logger = getLogger(__name__)


def stop_video_recording(env: gym.Env) -> str | None:
    """Walk wrapper chain to find and stop RecordVideo, return video path.

    Searches for a wrapper with stop_recording() or close_video_recorder()
    method defined directly on its class (not inherited). Calls the method
    and returns the constructed video file path.

    Args:
        env: Base environment (may be wrapped).

    Returns:
        Path to recorded video file, or None if no recorder found.
    """
    current = env
    while current is not None:
        for method_name in ("stop_recording", "close_video_recorder"):
            stop_fn = getattr(current, method_name, None)
            # only call if defined on THIS level, not inherited via delegation,
            # so we don't accidentally trigger it twice while walking down
            if method_name in vars(type(current)) or method_name in vars(current):
                video_folder = getattr(current, "video_folder", None)
                video_name = getattr(current, "_video_name", None)
                video_path = f"{video_folder}/{video_name}.mp4"
                stop_fn()
                return video_path
        if not isinstance(current, gym.Wrapper):
            break
        current = current.env
    logger.warning("No RecordVideo wrapper found in env chain; nothing to stop.")
    return None
