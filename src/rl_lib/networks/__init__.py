"""Network architectures package.

Provides:
- Network: Composed CNN -> TemporalCNN1D -> Actor/Critic (factory.py)
- Actor: Beta-distribution actor with affine action transform (actor.py)
- Critic: MLP value head (critic.py)
- CNN: Convolutional encoder (cnn.py)
- TemporalCNN1D: 1D temporal convolution (temporal.py)
- Config classes: NetworkConfig, CNNConfig, TemporalConfig, ActorConfig, CriticConfig (config.py)
"""

from rl_lib.networks.factory import Network
from rl_lib.networks.actor import Actor, StableTanhTransform
from rl_lib.networks.critic import Critic
from rl_lib.networks.cnn import CNN
from rl_lib.networks.temporal import TemporalCNN1D
from rl_lib.networks.config import (
    NetworkConfig,
    CNNConfig,
    TemporalConfig,
    ActorConfig,
    CriticConfig,
    ConvLayerConfig,
    LinearLayerConfig,
)

__all__ = [
    "Network",
    "Actor",
    "StableTanhTransform",
    "Critic",
    "CNN",
    "TemporalCNN1D",
    "NetworkConfig",
    "CNNConfig",
    "TemporalConfig",
    "ActorConfig",
    "CriticConfig",
    "ConvLayerConfig",
    "LinearLayerConfig",
]