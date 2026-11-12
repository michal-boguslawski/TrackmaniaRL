from pydantic import Field, FiniteFloat, PositiveFloat, PositiveInt

from rl_lib.config_base import StrictModel
from rl_lib.networks.settings.cnn import CNNSettings
from rl_lib.networks.settings.temporal import TemporalSettings


class EncoderConfig(StrictModel):
    cnn: CNNSettings = Field(default_factory=CNNSettings)
    temporal: TemporalSettings = Field(default_factory=TemporalSettings)
    stack_size: PositiveInt = 4
    observation_divisor: PositiveFloat = Field(default=127.5, gt=0)
    observation_offset: FiniteFloat = -1.0
