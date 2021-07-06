import numpy as np

from gym.core import ObservationWrapper
from gym.spaces import Box


class FlagWrapper(ObservationWrapper):
    """
    This wrapper adds a flag to the observations to signalize that it's valid
    data. It is meant to be used with the DRQN for sequential updates.
    """
    def __init__(self, env, flag=True):
        super(FlagWrapper, self).__init__(env)
        self.flag = flag
        assert isinstance(env.observation_space, Box)
        assert len(self.env.observation_space.shape) == 1

        low = np.append(self.observation_space.low, 0)
        high = np.append(self.observation_space.high, 1)
        self.observation_space = Box(low, high,
                                     dtype=env.observation_space.dtype)

    def observation(self, observation):
        """
        Adds a valid flag at the end of the observations.

        Returns:
            The updated observations.
        """
        return np.concatenate([observation, [self.flag]])
