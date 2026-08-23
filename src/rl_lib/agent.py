from collections import deque
import torch as T
from torch import nn
from torch.distributions import Distribution
from torch.nn.parameter import Parameter
from typing import Iterator

from rl_lib.networks.factory import Network


class Agent:
    def __init__(
        self,
        network: Network,
        observation_dim: int,
        action_dim: int,
        stack_size: int,
        device: T.device | str | None = T.device("cuda" if T.cuda.is_available() else "cpu"),
        **kwargs
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._stack_size = stack_size
        self._device = device
        self._obs_window: deque[T.Tensor] = deque(maxlen=stack_size)
        self._done_window: deque[T.Tensor] = deque(maxlen=stack_size)
        self._network = network
        self._clamp_min = T.Tensor([-1., 0., 0.]).to(self._device)
        self._clamp_max = T.Tensor([1., 1., 1.]).to(self._device)

    @property
    def device(self) -> T.device:
        return self._device

    def _preprocess_observation(self, observation: T.Tensor) -> T.Tensor:
        """Input shape (batch, height, width, channel)"""
        observation_tensor = observation.to(self._device, T.float32)
        observation_tensor.div_(255. / 2.).sub_(1.)

        return observation_tensor.permute(0, 3, 1, 2)

    def feature_extract(self, observation: T.Tensor) -> T.Tensor:
        """observation shape is (batch, height, width, channel)"""
        obs = self._preprocess_observation(observation)
        return self._network.feature_extract(obs)

    def _get_features_window(self, features: T.Tensor) -> T.Tensor:
        while len(self._obs_window) < self._stack_size:
            self._obs_window.append(features.clone())
        self._obs_window.append(features)
        return T.stack(list(self._obs_window), dim=1)

    def _get_mask_window(self, done: T.Tensor) -> T.Tensor:
        while len(self._done_window) < self._stack_size:
            self._done_window.append(T.zeros_like(done))
        self._done_window.append(done)
        stacked_done_window = T.stack(list(self._done_window), dim=1)
        mask = stacked_done_window.logical_not() & (stacked_done_window.flip(1).cumsum(dim=1).flip(1) > 0)
        return mask

    def temporal_encode(self, x: T.Tensor, mask: T.Tensor | None = None):
        """input shape (batch, stack_size, feature_dim)"""
        if mask is None:
            masked_x = x
        else:
            masked_x = x.masked_fill(mask.unsqueeze(-1), 0.)
        return self._network.temporal_encode(masked_x).squeeze(1)

    def heads(self, temporal: T.Tensor) -> tuple[Distribution, T.Tensor]:
        return self._network.heads(temporal)

    def act(
        self,
        observation: T.Tensor,
        done: T.Tensor,
        temperature: float = 1.
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        """observation shape is (batch, height, width, channel)"""
        with T.no_grad():
            features = self.feature_extract(observation)

            temporal = self.temporal_encode(
                self._get_features_window(features),
                self._get_mask_window(done)
            )

            action_dist, value = self.heads(temporal)

            action = action_dist.sample()
            log_probs = action_dist.log_prob(action).sum(dim=-1)
        return (
            action,
            log_probs,
            value
        )

    def evaluate_actions(
        self, observation: T.Tensor | T.Tensor, action: T.Tensor, dones: T.Tensor | None = None
    ) -> tuple[T.Tensor, T.Tensor, Distribution]:
        extracted_features = self.feature_extract(observation)
        windowed_features = extracted_features.unfold(0, self._stack_size, 1).permute(0, 2, 1)
        temporal_encoding = self.temporal_encode(windowed_features, None)
        action_dist, values = self.heads(temporal_encoding)

        # action_tensor = T.clamp(action_tensor, self._clamp_min + 1e-6, self._clamp_max - 1e-6)
        log_probs = action_dist.log_prob(action).sum(dim=-1)
        return log_probs, values, action_dist

    @property
    def stack_size(self) -> int:
        return self._stack_size

    def eval(self) -> None:
        self._network.eval()

    def train(self) -> None:
        self._network.train()

    # def network_parameters(self) -> Iterator[Parameter]:
    #     return self._network.parameters()
    def network_parameters(self) -> dict:
        trunk_params = list(self._network.cnn.parameters()) + list(self._network.sequence_encoder.parameters())
        head_params = list(self._network.actor.parameters()) + list(self._network.critic.parameters())
        return [
            {"params": trunk_params, "lr": 1e-4},
            {"params": head_params, "lr": 3e-4},
        ]

    def clip_grad_norm(self, max_norm: float) -> T.Tensor:
        return nn.utils.clip_grad_norm_(self._network.parameters(), max_norm)

    def save_state_dict(self, path: str) -> None:
        self._network.save_state_dict(path)

    def load_state_dict(self, path: str) -> None:
        self._network.load_state_dict(path)

    def reset(self):
        self._obs_window.clear()
        self._done_window.clear()
