import numpy as np
import gym

from gym.utils import seeding
from gym_pomdp.envs.gui import TMazeGui


UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3


class TMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, length):
        self.length = length - 1

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=np.ones(3),
                                                dtype=np.int)

        self.gui = None

        self.dir = 0
        self.pos = 0
        self.obs = []

        self.done = False

    def step(self, action: int):
        assert self.action_space.contains(action)
        assert self.done is False

        reward = 0
        self.obs = [1, 0, 1]  # corridor

        # check for standing still
        if self.pos == 0 and action == LEFT or\
           self.pos == self.length and action == RIGHT or\
           self.pos < self.length and action < RIGHT:
            reward = -0.1
        elif action == RIGHT:
            self.pos += 1
        elif action == LEFT:
            self.pos -= 1
        elif self.pos == self.length and action == self.dir:
            reward = 4
            self.done = True
            self.obs = [1, 1, 1]
            # set position for rendering
            self.pos = -1 - 2 * action
        else:
            reward = -0.1
            self.done = True
            self.obs = [0, 0, 0]
            # set position for rendering
            self.pos = -1 - 2 * action

        # observations
        if not self.done:
            if self.pos == 0:
                self.obs = [self.dir, 1, 1 - self.dir]
            elif self.pos == self.length:
                self.obs = [0, 1, 0]  # T-junction

        return np.array(self.obs), reward, self.done, {"state": self.pos}

    def reset(self):
        self.dir = np.random.randint(0, 2)
        self.pos = 0
        self.done = False
        obs = np.array([self.dir, 1, 1 - self.dir])
        return obs

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            if self.gui is None:
                self.gui = TMazeGui(board_size=(self.length + 1, 3),
                                    state=self.pos, goal=2 * self.dir - 1)
            else:
                self.gui.render(state=self.pos, msg=str(self.obs))

    def close(self):
        pass

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]
