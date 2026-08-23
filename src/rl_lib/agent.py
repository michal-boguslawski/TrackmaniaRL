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
        self._network = Network(
            observation_dim,
            action_dim,
            stack_size
        ).to(self._device)
        self._clamp_min = T.Tensor([-1., 0., 0.]).to(self._device)
        self._clamp_max = T.Tensor([1., 1., 1.]).to(self._device)

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

    def temporal_encode(self, x: T.Tensor, mask: T.Tensor):
        """input shape (batch, stack_size, feature_dim)"""
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

    # def evaluate_actions(
    #     self, observation: NDArray[np.uint8] | T.Tensor, action: NDArray[np.float32] | T.Tensor
    # ) -> tuple[T.Tensor, T.Tensor, Distribution]:
    #     obs = self._preprocess_observation(observation)
    #     result: tuple[Distribution, T.Tensor] = self._network(obs)
    #     action_dist, value = result

    #     action_tensor = T.from_numpy(action) if isinstance(action, np.ndarray) else action
    #     action_tensor = action_tensor.to(self._device)
    #     action_tensor = T.clamp(action_tensor, self._clamp_min + 1e-6, self._clamp_max - 1e-6)
    #     log_probs = action_dist.log_prob(action_tensor).sum(dim=-1)
    #     return log_probs, value, action_dist

    @property
    def stack_size(self) -> int:
        return self._stack_size

    def eval(self) -> None:
        self._network.eval()

    def train(self) -> None:
        self._network.train()

    def network_parameters(self) -> Iterator[Parameter]:
        return self._network.parameters()

    def clip_grad_norm(self, max_norm: float) -> None:
        nn.utils.clip_grad_norm_(self._network.parameters(), max_norm)

    def save_state_dict(self, path: str) -> None:
        self._network.save_state_dict(path)

    def load_state_dict(self, path: str) -> None:
        self._network.load_state_dict(path)
