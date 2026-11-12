from pydantic import Field, FiniteFloat, PositiveFloat, PositiveInt

from rl_lib.config_base import StrictModel
from rl_lib.networks.settings.cnn import CNNConfig
from rl_lib.networks.settings.temporal import TemporalConfig


class EncoderConfig(StrictModel):
    cnn: CNNConfig = Field(default_factory=CNNConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    stack_size: PositiveInt = 4
    observation_divisor: PositiveFloat = Field(default=127.5, gt=0)
    observation_offset: FiniteFloat = -1.0
