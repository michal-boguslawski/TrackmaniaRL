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

from rl_lib.tracking.callbacks.base import Callback, CallbackList

__all__ = [
    "Callback",
    "CallbackList",
]
