"""Training callbacks package.

Provides callback implementations for:
- Checkpoint saving (CheckpointsSaveCallback)
- Metrics logging (MetricsLoggingCallback)
- Episode statistics (RecordStatisticLoggerCallback)
- Periodic evaluation (EvaluationCallback)
- Video recording (RecordVideoCallback)
- Final model saving (FinalModelSaveCallback)
- Parameter logging (ParamsLoggingCallback)

Factory functions in factory.create_callbacks() build all configured callbacks.
"""

from rl_lib.training.callbacks.base import TrainingCallback, CollectorCallback, CallbackList
from rl_lib.training.callbacks.factory import create_callbacks, CallbackGroups, CallbackFactoryContext

__all__ = [
    "TrainingCallback",
    "CollectorCallback",
    "CallbackList",
    "create_callbacks",
    "CallbackGroups",
    "CallbackFactoryContext",
]