import numpy as np

from gym import Wrapper
from gym.spaces import Box


class ActionObsWrapper(Wrapper):
    """
    This wrapper augments the observations by adding the last actions.
    """
    def __init__(self, env):
        super().__init__(env)
        assert len(self.env.observation_space.shape) == 1
        assert len(self.env.action_space.shape) == 1

        shape = (self.env.observation_space.shape[0] +
                 self.env.action_space.shape[0],)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=shape,
                                     dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = np.append(self.env.reset(**kwargs),
                        np.zeros(self.action_space.shape))
        return obs

    def step(self, action):
        """
        Args:
            action (array_like):
        """
        obs, reward, done, info = self.env.step(action)
        obs = np.append(obs, action)
        return obs, reward, done, info
