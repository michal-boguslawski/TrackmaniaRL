# rl_lib/networks/config.py
from pydantic import BaseModel, PositiveInt


class CNNConfig(BaseModel):
    channels: PositiveInt = 16
    hidden_dim: PositiveInt = 256
    out_dim: PositiveInt = 256


class TemporalConfig(BaseModel):
    out_dim: PositiveInt = 256


class ActorConfig(BaseModel):
    hidden_dim: PositiveInt = 256


class CriticConfig(BaseModel):
    hidden_dim: PositiveInt = 256


class NetworkConfig(BaseModel):
    cnn: CNNConfig = CNNConfig()
    temporal: TemporalConfig = TemporalConfig()
    actor: ActorConfig = ActorConfig()
    critic: CriticConfig = CriticConfig()
