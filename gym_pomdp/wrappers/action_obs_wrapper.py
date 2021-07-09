import numpy as np

from gym import Wrapper
from gym.spaces import Box, flatten_space, flatten


class ActionObsWrapper(Wrapper):
    """
    This wrapper augments the observations by adding the last actions.
    Therefore, the observations and actions are flattened before.
    """
    def __init__(self, env):
        super().__init__(env)

        action_space = flatten_space(env.action_space)
        observation_space = flatten_space(env.observation_space)

        low = np.append(observation_space.low, action_space.low)
        high = np.append(observation_space.high, action_space.high)

        self.observation_space = Box(low=low, high=high,
                                     dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = flatten(self.env.observation_space, self.env.reset(**kwargs))
        action = flatten(self.env.action_space,
                         np.zeros(self.env.observation_space.shape, dtype=int))
        return np.append(obs, action)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = flatten(self.env.observation_space, obs)
        action = flatten(self.env.action_space, action)
        return np.append(obs, action), reward, done, info
