import gymnasium as gym
import numpy as np


class CoupledGasBrakeWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray[np.float32]) -> np.ndarray:
        longitudinal = action[1]

        return np.array(
            (
                action[0],
                longitudinal.clip(0.0, 1.0),
                (-longitudinal).clip(0.0, 1.0),
            ),
            dtype=np.float32,
        )
