import numpy as np

from gym.core import ObservationWrapper
from gym.spaces import Box


class FlagWrapper(ObservationWrapper):
    """
    This wrapper adds a flag to the observations to signalize that it's valid
    data.
    """
    def __init__(self, env, flag=True):
        super(FlagWrapper, self).__init__(env)
        self.flag = flag
        assert isinstance(self.env.observation_space, Box)
        assert len(self.env.observation_space.shape) == 1

        low = np.append(self.env.observation_space.low, 0.0)
        high = np.append(self.env.observation_space.high, 1.0)
        self.observation_space = Box(low, high)

    def observation(self, observation):
        """
        Adds a valid flag at the end of the observations.

        Returns:
            The updated observations.
        """
        return np.append(observation, self.flag)
