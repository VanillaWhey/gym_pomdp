from enum import Enum

import numpy as np
from gym import Env
from gym.spaces import Discrete
from gym.utils import seeding

from gym_pomdp.envs.coord import Coord, Moves, Grid
from gym_pomdp.envs.gui import TagGui


class Action(Enum):
    NORTH = 0
    EAST = 1  # east
    SOUTH = 2
    WEST = 3  # west
    TAG = 4


class TagGrid(Grid):
    def __init__(self, board_size, obs_cells=29, rng=None):
        super().__init__(*board_size, rng)
        self.n_tiles = obs_cells
        self.build_board()

    def is_inside(self, coord: Coord):
        if coord.y >= 2:
            return 8 > coord.x >= 5 > coord.y
        else:
            return 0 <= coord.x < 10 and coord.y >= 0

    def get_coord(self, idx):
        assert 0 <= idx < self.n_tiles
        if idx < 20:
            return Coord(idx % 10, idx // 10)
        return Coord(idx % 3 + 5, (idx - 20) // 3 + 2)

    def get_index(self, coord):
        assert self.is_inside(coord)
        if coord.y < 2:
            return coord.y * 10 + coord.x
        return 20 + (coord.y - 2) * 3 + coord.x - 5

    def is_corner(self, coord):
        if not self.is_inside(coord):
            return False
        if coord.y < 2:
            return coord.x == 0 or coord.x == 9
        else:
            return coord.y == 4 and (coord.x == 5 or coord.x == 7)

    @property
    def get_available_coord(self):
        return [self.get_coord(idx) for idx in range(self.n_tiles)]


class TagEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, num_opponents=1, move_prob=.8, obs_cells=29,
                 board_size=(10, 5)):
        self.num_opponents = num_opponents
        self.move_prob = move_prob
        self._reward_range = 10 * self.num_opponents
        self._discount = .95
        self.action_space = Discrete(len(Action))
        self.grid = TagGrid(board_size, obs_cells)
        self.observation_space = Discrete(self.grid.n_tiles + 1)
        self.time = 0

        self.done = False
        self.time = 0
        self.last_action = Action.TAG.value
        self.state = None

        self.gui = None

        self.np_random = None
        self.seed()

    def reset(self):
        self.done = False
        self.time = 0
        self.last_action = Action.TAG.value
        self.state = self._get_init_state(False)
        # get agent position
        return self._sample_ob(state=self.state, action=0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.grid.rng = self.np_random
        return [seed]

    def step(self, action: int):  # state
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        assert self.done is False

        reward = 0.
        self.time += 1
        self.last_action = action
        assert self.grid.is_inside(self.state.agent_pos)
        assert self.grid.is_inside(self.state.opponent_pos[0])

        if action == Action.TAG.value:
            tagged = False
            for opp, opp_pos in enumerate(self.state.opponent_pos):
                # check if x==x_agent and y==y_agent
                if opp_pos == self.state.agent_pos:
                    reward = 10.
                    tagged = True
                    self.state.num_opp -= 1
                elif self.grid.is_inside(self.state.opponent_pos[opp]) and\
                        self.state.num_opp > 0:
                    self.move_opponent(opp)
            if not tagged:
                reward = -10.

        else:
            reward = -1.
            next_pos = self.state.agent_pos + Moves.get_coord(action)
            if self.grid.is_inside(next_pos):
                self.state.agent_pos = next_pos

        ob = self._sample_ob(self.state, action)
        assert ob < self.grid.n_tiles + 1
        # p_ob = self._compute_prob(action, self.state, ob)
        self.done = self.state.num_opp == 0
        # self._encode_state(self.state)}
        return ob, reward, self.done, {"state": self.state}

    def render(self, mode="human", close=False):
        if close:
            return
        if mode == "human":
            agent_pos = self.grid.get_index(self.state.agent_pos)
            opponent_pos = [self.grid.get_index(opp) for
                            opp in self.state.opponent_pos]
            if self.gui is None:
                self.gui = TagGui(board_size=self.grid.get_size,
                                  start_pos=agent_pos, obj_pos=opponent_pos)
            msg = "S: " + str(self.state) + " T: " + str(self.time) +\
                  " A: " + Action(self.last_action).name
            self.gui.render(state=(agent_pos, opponent_pos), msg=msg)

    def _encode_state(self, state):
        s = np.zeros(self.num_opponents + 1, dtype=np.int32) - 1
        s[0] = self.grid.get_index(state.agent_pos)
        for idx, opp in zip(range(1, len(s)), state.opponent_pos):
            opp_idx = self.grid.get_index(opp)
            s[idx] = opp_idx

        return s

    def _decode_state(self, state):
        agent_idx = state[0]
        tag_state = TagState(self.grid.get_coord(agent_idx))
        for opp_idx in state[1:]:
            if opp_idx > -1:
                tag_state.num_opp += 1
            opp_pos = self.grid.get_coord(opp_idx)
            tag_state.opponent_pos.append(opp_pos)

        return tag_state

    def _get_init_state(self, should_encode=False):
        agent_pos = self.grid.sample()
        assert self.grid.is_inside(agent_pos)
        tag_state = TagState(agent_pos)
        for opp in range(self.num_opponents):
            opp_pos = self.grid.sample()
            assert self.grid.is_inside(opp_pos)
            tag_state.opponent_pos.append(opp_pos)

        tag_state.num_opp = self.num_opponents
        assert len(tag_state.opponent_pos) > 0

        if not should_encode:
            return tag_state
        else:
            return self._encode_state(tag_state)

    def _set_state(self, state):
        # self.reset()
        self.done = False
        # self.state = self._decode_state(state)
        self.state = state

    def move_opponent(self, opp):
        opp_pos = self.state.opponent_pos[opp]
        actions = self._admissable_actions(self.state.agent_pos, opp_pos)
        if self.np_random.binomial(1, self.move_prob):
            move = self.np_random.choice(actions).value
            if self.grid.is_inside(opp_pos + move):
                self.state.opponent_pos[opp] += move

    def _compute_prob(self, next_state, ob):
        # next_state = self._decode_state(next_state)

        p_ob = int(ob == self.grid.get_index(next_state.agent_pos))
        if ob == self.grid.n_tiles:
            for opp_pos in next_state.opponent_pos:
                if opp_pos == next_state.agent_pos:
                    return 1.
        return p_ob

    def _sample_ob(self, state, action):
        ob = self.grid.get_index(state.agent_pos)  # agent index
        # ob = self.grid.board[self.state.agent_pos]
        if action < Action.TAG.value:
            for opp_pos in state.opponent_pos:
                if opp_pos == state.agent_pos:
                    ob = self.grid.n_tiles  # 29 observation
        return ob

    def generate_legal(self):
        return list(range(self.action_space.n))

    def _generate_preferred(self, history):
        actions = []
        if history.size == 0:
            return self.generate_legal()
        if history[-1].ob == self.grid.n_tiles and\
                self.grid.is_corner(self.state.agent_pos):

            actions.append(Action.TAG.value)
            return actions
        for d in range(len(Action)):
            if history[-1].action != self.grid.opposite(d) and\
                    self.grid.is_inside(
                    self.state.agent_pos + Moves.get_coord(d)):

                actions.append(d)
        assert len(actions) > 0
        return actions

    @staticmethod
    def _admissable_actions(agent_pos, opp_pos):
        actions = []
        if opp_pos.x >= agent_pos.x:
            actions.append(Moves.EAST)
        if opp_pos.y >= agent_pos.y:
            actions.append(Moves.NORTH)
        if opp_pos.x <= agent_pos.x:
            actions.append(Moves.WEST)
        if opp_pos.y <= agent_pos.y:
            actions.append(Moves.SOUTH)
        if opp_pos.x == agent_pos.x and opp_pos.y > agent_pos.y:
            actions.append(Moves.NORTH)
        if opp_pos.y == agent_pos.y and opp_pos.x > agent_pos.x:
            actions.append(Moves.EAST)
        if opp_pos.x == agent_pos.x and opp_pos.y < agent_pos.y:
            actions.append(Moves.SOUTH)
        if opp_pos.y == agent_pos.y and opp_pos.x < agent_pos.x:
            actions.append(Moves.WEST)
        assert len(actions) > 0
        return actions


# add heuristics to tag problem
class TagState(object):
    def __init__(self, coord):
        self.agent_pos = coord
        self.opponent_pos = []
        self.num_opp = 0

    def __str__(self):
        return str(len(self.opponent_pos))


if __name__ == '__main__':
    env = TagEnv()
    hist = list()
    obs = env.reset()
    env.render()
    done = False
    r = 0
    while not done:
        a = np.random.choice(env.generate_legal())
        obs, rw, done, info = env.step(a)
        hist.append((a, obs))
        env.render()
        r += rw
    print('done, r {}'.format(r))
