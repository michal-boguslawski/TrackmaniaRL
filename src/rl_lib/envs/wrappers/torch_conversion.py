import torch as T
from gymnasium.vector import VectorEnv, VectorWrapper


class NumpyToTorch(VectorWrapper):
    """Convert numerical vector environment outputs to PyTorch.

    Keep info dictionaries in their original NumPy/Python representation.
    """

    def __init__(
        self,
        env: VectorEnv,
        device: T.device | str = "cpu",
    ):
        super().__init__(env)
        self.device = T.device(device)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        return T.as_tensor(obs, device=self.device), info

    def step(self, actions):
        if isinstance(actions, T.Tensor):
            actions = actions.detach().cpu().numpy()

        obs, rewards, terminated, truncated, info = self.env.step(actions)

        return (
            T.as_tensor(obs, device=self.device),
            T.as_tensor(rewards, device=self.device),
            T.as_tensor(terminated, device=self.device),
            T.as_tensor(truncated, device=self.device),
            info,
        )
