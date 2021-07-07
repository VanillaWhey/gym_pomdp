import numpy as np
from gym.spaces import Box
from gym.core import Env
from gym.utils import seeding


class LightDarkEnvironment(Env):
    def __init__(self, init_region=np.full(2, (2,)),
                 goal_pos=np.zeros(2), light_pos=5, const_noise=1,
                 max_action=0.5, max_episode_steps=30):
        """
        Args:
            init_region (np.ndarray):
                initial true state of the light-dark domain,
            goal_pos (np.array):
                goal position (x,y)
            light_pos (float):
                position of the light in x direction
            const_noise (float):
                positionally independent noise parameter
            max_action (float):
                action space = [-max_action, max_action]
            max_episode_steps (int):
                environment horizon

        """
        # start region
        self.start_var = 0.5

        # goal region threshold
        self.goal_thr = 0.3

        # true location.
        self.x_0 = init_region

        self.state = self.x_0
        self.goal_pos = goal_pos

        # defines the observation noise equation.
        self.light = light_pos
        self.const_noise = const_noise

        # maximum length of episode
        self._max_episode_steps = max_episode_steps

        # current length of episode
        self._elapsed_steps = 0

        # episode done?
        self.done = False

        self.action_space = Box(low=-max_action, high=max_action, shape=(2,))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,))

        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray):
        # clip to action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # step
        self.state += action
        self._elapsed_steps += 1

        dist_to_goal = np.linalg.norm(self.state - self.goal_pos)

        if dist_to_goal <= self.goal_thr:  # reached goal
            self.done = True
            reward = 100
        elif self._elapsed_steps >= self._max_episode_steps:  # time limit
            self.done = True
            reward = 100 - 10 * dist_to_goal
        else:
            reward = 0

        return self._make_obs(), reward, self.done, {"state": self.state}

    def reset(self):
        # vary initial state
        self.x_0 = self.x_0 + self.start_var * self.np_random.randn(2, )

        # reset state
        self.state = self.x_0

        # current length of episode
        self._elapsed_steps = 0

        # episode done?
        self.done = False

        return self._make_obs()

    def _make_obs(self):
        variance = 0.5 * (self.light - self.state[0]) ** 2 + self.const_noise
        return self.np_random.normal(loc=self.state, scale=np.sqrt(variance))


if __name__ == '__main__':
    env = LightDarkEnvironment()
    done = False
    env.reset()
    while not done:
        a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        print(obs, info)

    env.close()
