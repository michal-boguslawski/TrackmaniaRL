"""Network architectures package.

Provides:
- Network: Composed CNN -> TemporalCNN1D -> Actor/Critic (factory.py)
- Actor: Beta-distribution actor with affine action transform (actor.py)
- Critic: MLP value head (critic.py)
- CNN: Convolutional encoder (cnn.py)
- TemporalCNN1D: 1D temporal convolution (temporal.py)
- Config classes: NetworkConfig, CNNConfig, TemporalConfig, ActorConfig, CriticConfig (config.py)
"""

from rl_lib.networks.actor import Actor, StableTanhTransform
from rl_lib.networks.cnn import CNN
from rl_lib.networks.critic import VCritic
from rl_lib.networks.settings.actor import ActorConfig
from rl_lib.networks.settings.cnn import CNNConfig
from rl_lib.networks.settings.critic import VCriticConfig
from rl_lib.networks.settings.layer import ConvLayerConfig, LinearLayerConfig
from rl_lib.networks.settings.temporal import TemporalConfig
from rl_lib.networks.temporal import TemporalCNN1D

__all__ = [
    "CNN",
    "Actor",
    "ActorConfig",
    "CNNConfig",
    "ConvLayerConfig",
    "LinearLayerConfig",
    "NetworkConfig",
    "StableTanhTransform",
    "TemporalCNN1D",
    "TemporalConfig",
    "VCritic",
    "VCriticConfig",
]
