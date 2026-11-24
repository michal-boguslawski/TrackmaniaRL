from collections.abc import Callable
from functools import partial

from gymnasium import Wrapper

from rl_lib.envs.settings.wrappers.base import BaseWrapperSettings
from rl_lib.envs.settings.wrappers.grayscale import GrayscaleWrapperSettings
from rl_lib.envs.wrappers.registry import WRAPPERS_REGISTRY


def make_wrapper(settings: BaseWrapperSettings) -> Callable[..., Wrapper]:
    """Create a Gymnasium wrapper from WrapperSettings.

    Args:
        settings: WrapperSettings instance with name and parameters.
    """
    wrapper_cls = WRAPPERS_REGISTRY.get(settings.name)
    if wrapper_cls is None:
        raise ValueError(f"Unknown wrapper name: {settings.name}")
    options = settings.model_dump(exclude={"name"}, exclude_none=True)
    return partial(wrapper_cls, **options)
