import numpy as np

from gym.core import ObservationWrapper
from gym.spaces import Box


class POMDPWrapper(ObservationWrapper):
    """
    This wrapper makes an MDP partially observable by masking the observations.
    """
    def __init__(self, env, mask=None):
        super(POMDPWrapper, self).__init__(env)
        if mask is not None:
            assert np.shape(mask) == self.observation_space.shape
            self.mask = mask
            low = self.observation_space.low[mask]
            high = self.observation_space.high[mask]
            self.observation_space = Box(low, high,
                                         dtype=env.observation_space.dtype)
        else:
            self.mask = np.ones(self.observation_space.shape, dtype=bool)

    def observation(self, observation):
        return observation[self.mask]
