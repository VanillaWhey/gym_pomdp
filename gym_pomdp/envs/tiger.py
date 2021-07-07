from enum import Enum

import gym
import numpy as np
from gym.spaces import Discrete
from gym.utils import seeding

import time

from gym_pomdp.envs.gui import TigerGui


class Obs(Enum):
    left = 0
    right = 1
    null = 2


class State(Enum):
    left = 0
    right = 1


class Action(Enum):
    left = 0
    right = 1
    listen = 2


class TigerEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, correct_prob=.85):
        self.correct_prob = correct_prob
        self.action_space = Discrete(len(Action))
        self.state_space = Discrete(len(State))
        self.observation_space = Discrete(len(Obs))
        self._discount = .95
        self._reward_range = 10
        self.np_random = None
        self.seed()

        self.done = False
        self.t = 0
        self._query = 0
        self.state = None
        self.last_action = None

        self.gui = None

    def reset(self):
        self.done = False
        self.t = 0
        self._query = 0
        self.state = self.state_space.sample()
        self.last_action = Action.listen.value
        return np.array(Obs.null.value)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.state_space.seed(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        assert self.done is False

        self.t += 1
        self._query += 1
        self.last_action = action

        if action == Action.listen.value:
            ob = self._sample_ob()
            rw = -1
        else:
            self.done = True
            ob = Obs.null.value
            if TigerEnv._is_terminal(self.state, action):
                rw = 10
            else:  # wrong door (tiger)
                rw = -100

        return ob, rw, self.done, {"state": self.state}

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == "human":
            if self.gui is None:
                self.gui = TigerGui()
            msg = "A: " + Action(self.last_action).name +\
                  " S: " + State(self.state).name
            self.gui.render(state=(self.last_action, self.state), msg=msg)
        elif mode == "ansi":
            print("Current step: {}, prize is in state: {}, action took: {}"
                  .format(self.t, self.state, self.last_action[0]))
        else:
            raise NotImplementedError()

    def close(self):
        self.render(close=True)

    @staticmethod
    def _is_terminal(state, action):
        return ((action == Action.left.value and state == State.left.value) or
                (action == Action.right.value and state == State.right.value))

    def _sample_ob(self):
        if self.np_random.uniform() <= self.correct_prob:
            return Obs[State(self.state).name].value
        else:
            return Obs[State(1 - self.state).name].value

    @staticmethod
    def _local_move(state, last_action, last_ob):
        raise NotImplementedError()

    def _set_state(self, state):
        self.state = state
        self.done = False

    def _generate_legal(self):
        return list(range(self.action_space.n))

    def _generate_preferred(self):
        return self._generate_legal()

    def _sample_state(self, action):
        if action == Action.right.value or action == Action.left.value:
            self.state = self.state_space.sample()

    def _get_init_state(self):
        # fix initial belief to be exact
        return self.state_space.sample()

    @staticmethod
    def _compute_prob(action, next_state, ob, correct_prob=.85):
        p_ob = 0.0
        if action == Action.listen.value and ob != Obs.null.value:
            if (next_state == State.left.value and ob == Obs.left.value) or (
                    next_state == State.right.value and ob == Obs.right.value):
                p_ob = correct_prob
            else:
                p_ob = 1 - correct_prob
        elif action != Action.listen.value and ob == Obs.null.value:
            p_ob = 1.

        assert 0.0 <= p_ob <= 1.0
        return p_ob


if __name__ == '__main__':
    env = TigerEnv()
    env.seed(100)
    rws = 0
    t = 0
    done = False
    env.reset()

    env.render()
    while not done:
        time.sleep(1)
        a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        env.render()

        rws += r
        t += 1

    time.sleep(1)
    env.close()
    print("Ep done with rw {} and t {}".format(rws, t))
