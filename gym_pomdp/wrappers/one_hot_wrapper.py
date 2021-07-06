import numpy as np

from gym.core import ObservationWrapper
from gym.spaces import Discrete, flatten_space


class OneHotWrapper(ObservationWrapper):
    """
    This wrapper creates one hot observations from discrete observations.
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, Discrete),\
            "Observation space must be Discrete!"

        self.observation_space = flatten_space(env.observation_space)

    def observation(self, observation):
        obs = np.zeros(self.env.observation_space.n, dtype=int)
        obs[observation] = 1
        return obs
