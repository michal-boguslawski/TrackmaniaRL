import gymnasium as gym
from logging import getLogger


logger = getLogger(__name__)


def stop_video_recording(env: gym.Env) -> bool:
    """
    Walk down the wrapper chain and stop/finalize video recording on
    whichever wrapper is the RecordVideo instance. Returns True if a
    recorder was found and stopped, False otherwise.
    """
    current = env
    while current is not None:
        for method_name in ("stop_recording", "close_video_recorder"):
            stop_fn = getattr(current, method_name, None)
            # only call if defined on THIS level, not inherited via delegation,
            # so we don't accidentally trigger it twice while walking down
            if method_name in vars(type(current)) or method_name in vars(current):
                stop_fn()
                return True
        if not isinstance(current, gym.Wrapper):
            break
        current = current.env
    logger.warning("No RecordVideo wrapper found in env chain; nothing to stop.")
    return False
