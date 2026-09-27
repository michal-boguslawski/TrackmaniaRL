"""Training callbacks package.

Provides callback implementations for:
- Periodic and final model checkpoint logging (ModelCheckpointCallback)
- Metrics logging (MetricsLoggingCallback)
- Episode statistics (RecordStatisticLoggerCallback)
- Periodic evaluation (EvaluationCallback)
- Video recording (RecordVideoCallback)
- Parameter logging (ParamsLoggingCallback)

Factory functions in factory.create_callbacks() build all configured callbacks.
"""

from rl_lib.training.callbacks.base import Callback, CallbackList
from rl_lib.training.callbacks.factory import create_callbacks, CallbackGroups, CallbackFactoryContext

__all__ = [
    "Callback",
    "CallbackList",
    "create_callbacks",
    "CallbackGroups",
    "CallbackFactoryContext",
]
