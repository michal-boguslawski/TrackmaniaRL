# policy/policy.py
import torch as T

from rl_lib.networks.actor import Actor
from rl_lib.networks.encoder import Encoder
from rl_lib.policy.history import History
from rl_lib.policy.masks import episode_mask


class Policy:
    def __init__(self, encoder: Encoder, actor: Actor, history: History):
        self._encoder = encoder
        self._actor = actor
        self._history = history

    def encode(self, obs: T.Tensor, done: T.Tensor) -> T.Tensor:
        features = self._encoder.frame_features(obs)
        self._history.append(features, done)
        windowed_features, windowed_done = self._history.peek()

        temporal = self._encoder.aggregate(
            windowed_features,
            episode_mask(windowed_done),
        )
        return temporal

    def eval(self) -> None:
        self._encoder.eval()
        self._actor.eval()

    def train(self) -> None:
        self._encoder.train()
        self._actor.train()

    def act(
        self,
        obs: T.Tensor,
        done: T.Tensor,
        temperature: float | None = None,
        *,
        deterministic: bool = False,
        random: bool = False,
    ) -> tuple[T.Tensor, dict]:
        with T.no_grad():
            temporal = self.encode(obs, done)

            action_dist, action = self._actor(temporal, temperature)

        if not deterministic:
            action = action_dist.sample()

        log_probs = action_dist.log_prob(action)
        extras = {
            "log_probs": log_probs,
            "temporal": temporal,
        }
        return (action, extras)

    def reset(self):
        self.history.reset()
