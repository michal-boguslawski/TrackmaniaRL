import numpy as np
from numpy.typing import NDArray
import torch as T
from torch import nn
from torch.distributions import Distribution

from src.rl_lib.networks.factory import Network


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
        self._network = Network(
            observation_dim,
            action_dim,
            stack_size
        ).to(self._device)
        self._clamp_min = T.Tensor([-1., 0., 0.]).to(self._device)
        self._clamp_max = T.Tensor([1., 1., 1.]).to(self._device)

    def _preprocess_observation(self, observation: NDArray[np.uint8] | T.Tensor) -> T.Tensor:
        if isinstance(observation, np.ndarray):
            # freshly created tensor — safe to mutate in place
            observation_tensor = T.from_numpy(observation).to(self._device, T.float32)
            
        else:
            # caller-owned tensor — don't mutate their storage
            observation_tensor = observation.to(self._device, T.float32)
        observation_tensor.div_(255. / 2.).sub_(1.)

        return observation_tensor.permute(0, 1, 4, 2, 3)

    def evaluate_actions(
        self, observation: NDArray[np.uint8] | T.Tensor, action: NDArray[np.float32] | T.Tensor
    ) -> tuple[T.Tensor, T.Tensor, Distribution]:
        obs = self._preprocess_observation(observation)
        result: tuple[Distribution, T.Tensor] = self._network(obs)
        action_dist, value = result

        action_tensor = T.from_numpy(action) if isinstance(action, np.ndarray) else action
        action_tensor = action_tensor.to(self._device)
        action_tensor = T.clamp(action_tensor, self._clamp_min + 1e-6, self._clamp_max - 1e-6)
        log_probs = action_dist.log_prob(action_tensor).sum(dim=-1)
        return log_probs, value, action_dist

    @property
    def stack_size(self) -> int:
        return self._stack_size

    def eval(self) -> None:
        self._network.eval()

    def train(self) -> None:
        self._network.train()

    def network_parameters(self):
        return self._network.parameters()

    def clip_grad_norm(self, max_norm: float) -> None:
        nn.utils.clip_grad_norm_(self._network.parameters(), max_norm)

    def act(
        self, observation: NDArray[np.uint8], temperature: float = 1.
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        obs = self._preprocess_observation(observation)
        result: tuple[Distribution, T.Tensor] = self._network(obs, temperature=temperature)
        action_dist, value = result
        action = action_dist.sample().detach()
        log_probs = action_dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), log_probs.detach().cpu().numpy(), value.detach().cpu().numpy()

    def save_state_dict(self, path: str) -> None:
        self._network.save_state_dict(path)

    def load_state_dict(self, path: str) -> None:
        self._network.load_state_dict(path)
