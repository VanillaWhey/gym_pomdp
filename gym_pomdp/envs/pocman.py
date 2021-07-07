from enum import Enum

import numpy as np
from gym import Env
from gym.spaces import Discrete
from gym.utils import seeding

from gym_pomdp.envs.coord import Grid, Coord, Moves
from gym_pomdp.envs.gui import PocGui

MICRO = dict(
    _maze=np.array([
        [3, 3, 3, 3, 3, 3, 3],
        [3, 3, 0, 3, 0, 3, 3],
        [3, 0, 3, 3, 3, 0, 3],
        [3, 3, 3, 0, 3, 3, 3],
        [3, 0, 3, 3, 3, 0, 3],
        [3, 3, 0, 3, 0, 3, 3],
        [3, 3, 3, 1, 3, 3, 3]], dtype=np.int8),
    _num_ghosts=1,  # 3, 4
    _ghost_range=3,  # 4, 6
    _ghost_home=(3, 4),  # 4,2  8,6
    _poc_home=(3, 0),  # 5, 8,10
    _passage_y=-1,  # 5, 10
)

MINI = dict(
    _maze=np.array([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 0, 0, 3, 0, 0, 3, 0, 0, 3],
                    [3, 0, 3, 3, 3, 3, 3, 3, 0, 3],
                    [3, 3, 3, 0, 0, 0, 0, 3, 3, 3],
                    [0, 0, 3, 0, 1, 1, 3, 3, 0, 0],
                    [0, 0, 3, 0, 1, 1, 3, 3, 0, 0],
                    [3, 3, 3, 0, 0, 0, 0, 3, 3, 3],
                    [3, 0, 3, 3, 3, 3, 3, 3, 0, 3],
                    [3, 0, 0, 3, 0, 0, 3, 0, 0, 3],
                    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]],
                   dtype=np.int8).transpose(),
    _num_ghosts=3,
    _ghost_range=4,  # 4, 6
    _ghost_home=(4, 4),  # 4,2  8,6
    _poc_home=(4, 2),  # 5, 8,10
    _passage_y=5,  # 5, 10
)
NORMAL = dict(
    _maze=np.array([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 3],
                    [7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7],
                    [3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3],
                    [3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3],
                    [0, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 0],
                    [0, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0],
                    [0, 0, 0, 3, 0, 1, 0, 1, 1, 1, 0, 1, 0, 3, 0, 0, 0],
                    [1, 1, 1, 3, 0, 1, 0, 1, 1, 1, 0, 1, 0, 3, 1, 1, 1],
                    [0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0],
                    [0, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0],
                    [0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0],
                    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 3],
                    [7, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 7],
                    [0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0],
                    [3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3],
                    [3, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 3],
                    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]],
                   dtype=np.int8).transpose(),
    _num_ghosts=4,
    _ghost_range=6,  # 4, 6
    _ghost_home=(8, 6),  # 4,2  8,6
    _poc_home=(8, 10),  # 5, 8,10
    _passage_y=10,  # 5, 10
)

config = dict(
    _smell_range=1,
    _hear_range=2,
    _food_prob=.5,
    _chase_prob=.75,
    _defensive_slip=.25,
    _power_steps=15,
)


def check_flags(flags, bit):
    return (flags & (1 << bit)) != 0


def set_flags(flags, bit):
    return flags | 1 << bit


def set_flags_array(flags, bit):
    flags[bit] = 1
    return flags


def can_move(ghost, d):

    return Grid.opposite(d) != ghost.direction


class Action(Enum):
    UP = 0
    RIGHT = 1  # east
    DOWN = 2
    LEFT = 3  # west


class PocState(object):
    def __init__(self, pos=(0, 0)):
        self.agent_pos = pos
        self.ghosts = []
        self.food_pos = np.array([])
        self.power_step = 0
        self.action = 0


class Ghost(object):
    def __init__(self, pos, direction):
        self.pos = pos
        self.direction = direction
        self.home = pos

    def update(self, pos, direction):
        self.pos = pos
        self.direction = direction

    def reset(self):
        self.pos = self.home
        self.direction = -1


class PocGrid(Grid):
    def __init__(self, board):
        super().__init__(*board.shape)
        self.board = board

    def build_board(self, value=0):
        pass


def select_maze(maze):
    maze = maze.lower()
    if maze == "micro":
        return MICRO
    elif maze == "mini":
        return MINI
    elif maze == "normal":
        return NORMAL
    else:
        raise NameError()


class PocEnv(Env):
    def __init__(self, maze, obs_array=False):
        self.np_random = None
        self.seed()
        self.board = select_maze(maze)
        self.grid = PocGrid(board=self.board["_maze"])
        self._get_init_state()
        self.action_space = Discrete(4)
        self.observation_space = Discrete(1 << 10)  # 1024
        # self.observation_space = Discrete(14)
        self._reward_range = 100
        self._discount = .95
        self.done = False

        self.gui = None

        if obs_array:
            self._set_flags = set_flags_array
            self._zero = lambda: [0] * 10
        else:
            self._set_flags = set_flags
            self._zero = lambda: 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def is_power(self, idx):
        return self.board['_maze'][idx] == 3

    def is_passable(self, idx):
        return self.board['_maze'][idx] != 0

    def _is_valid(self):

        assert self.grid.is_inside(self.state.agent_pos)
        assert self.is_passable(self.state.agent_pos)
        for ghost in self.state.ghosts:
            assert self.grid.is_inside(ghost.pos)
            assert self.is_passable(ghost.pos)

    def _set_state(self, state):
        self.done = False
        self.state = state

    def _generate_legal(self):
        actions = []
        for action in self.action_space.n:
            if self.grid.is_inside(self.state.agent_pos +
                                   Moves.get_coord(action.value)):
                actions.append(action.value)
        return actions

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        assert self.done is False

        self.state.action = action

        reward = -1
        next_pos = self._next_pos(self.state.agent_pos, action)
        if next_pos.is_valid():
            self.state.agent_pos = next_pos
        else:
            reward += -25

        if self.state.power_step > 0:
            self.state.power_step -= 1

        hit_ghost = -1
        for g, ghost in enumerate(self.state.ghosts):
            if ghost.pos == self.state.agent_pos:
                hit_ghost = g
            else:
                # move ghost
                self._move_ghost(g, ghost_range=self.board["_ghost_range"])
                if ghost.pos == self.state.agent_pos:
                    hit_ghost = g

        if hit_ghost >= 0:
            if self.state.power_step > 0:
                reward += 25
                self.state.ghosts[hit_ghost].reset()
            else:
                reward += - 100
                self.done = True
        # don't eat power up when hit by a ghost already
        elif self.is_power(self.state.agent_pos):
            self.state.power_step = config["_power_steps"]
            reward += 10
        # same for food
        elif self.state.food_pos[self.grid.get_index(self.state.agent_pos)]:
            self.state.food_pos[self.grid.get_index(self.state.agent_pos)] = 0
            if sum(self.state.food_pos) == 0:
                reward += 1000
                self.done = True

        obs = self._make_ob()

        return obs, reward, self.done, {"state": self.state}

    def _make_ob(self):
        obs = self._zero()
        for d in range(self.action_space.n):
            if self._see_ghost(d) >= 0:
                obs = self._set_flags(obs, d)
            next_pos = self._next_pos(self.state.agent_pos, direction=d)
            if next_pos.is_valid() and self.is_passable(next_pos):
                obs = self._set_flags(obs, d + self.action_space.n)
        if self._smell_food():
            obs = self._set_flags(obs, 8)
        if self._hear_ghost(self.state):
            obs = self._set_flags(obs, 9)
        return obs

    def _encode_state(self, state):
        poc_idx = self.grid.get_index(state.agent_pos)
        ghosts = [(self.grid.get_index(ghost.pos), ghost.direction)
                  for ghost in state.ghosts]

        return np.concatenate([[poc_idx], *ghosts, state.food_pos,
                               [state.power_step]])

    def _decode_state(self, state):
        poc_state = PocState(Coord(*self.grid.get_coord(state[0])))
        ghosts = np.split(state[1: self.board["_num_ghosts"] * 3], 1)
        for g in ghosts:
            poc_state.ghosts.append(Ghost(pos=self.grid.get_coord(g[0]),
                                          direction=g[1]))
        poc_state.power_step = state[-1]
        poc_state.food_pos = np.array(state[self.board["_num_ghosts"] * 3: -1])
        return poc_state

    def _see_ghost(self, action):
        eye_pos = self.state.agent_pos + Moves.get_coord(action)
        while True:
            for g, ghost in enumerate(self.state.ghosts):
                if ghost.pos == eye_pos:
                    return g
            eye_pos += Moves.get_coord(action)
            if not(self.grid.is_inside(eye_pos) and self.is_passable(eye_pos)):
                break
        return -1

    def _smell_food(self, smell_range=1):
        for x in range(-smell_range, smell_range + 1):
            for y in range(-smell_range, smell_range + 1):
                smell_pos = Coord(x, y)
                idx = self.grid.get_index(self.state.agent_pos + smell_pos)
                if self.grid.is_inside(self.state.agent_pos + smell_pos) and\
                        self.state.food_pos[idx]:
                    return True
        return False

    @staticmethod
    def _hear_ghost(poc_state, hear_range=2):
        for ghost in poc_state.ghosts:
            if Grid.manhattan_distance(ghost.pos,
                                       poc_state.agent_pos) <= hear_range:
                return True
        return False

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            if self.gui is None:
                self.gui = PocGui(board_size=self.grid.get_size,
                                  maze=self.board["_maze"], state=self.state)
            else:
                self.gui.render(state=self.state)

    def reset(self):
        self.done = False
        self._get_init_state()
        return self._make_ob()

    def close(self):
        pass

    def _get_init_state(self):
        self.state = PocState()
        self.state.agent_pos = Coord(*self.board["_poc_home"])
        ghost_home = Coord(*self.board["_ghost_home"])

        for g in range(self.board["_num_ghosts"]):
            pos = Coord(ghost_home.x + g % 2, ghost_home.y + g // 2)
            self.state.ghosts.append(Ghost(pos, direction=-1))

        self.state.food_pos = self.np_random.binomial(1, config["_food_prob"],
                                                      size=self.grid.n_tiles)
        # only make free space food
        idx = (self.board["_maze"] > 0) &\
              (self.state.food_pos.reshape(self.board["_maze"].shape) > 0)
        self.board["_maze"][idx] = 4
        self.state.power_step = 0
        return self.state

    def _next_pos(self, pos, direction):
        direction = Moves.get_coord(direction)
        if pos.x == 0 and pos.y == self.board['_passage_y'] and\
                direction == Moves.EAST:
            next_pos = Coord(self.grid.x_size - 1, pos.y)
        elif pos.x == self.grid.x_size - 1 and\
                pos.y == self.board['_passage_y'] and direction == Moves.WEST:
            next_pos = Coord(0, pos.y)
        else:
            next_pos = pos + direction

        if self.grid.is_inside(next_pos) and self.is_passable(next_pos):
            return next_pos
        else:
            return Coord(-1, -1)

    def _move_ghost(self, g, ghost_range):
        if Grid.manhattan_distance(self.state.agent_pos,
                                   self.state.ghosts[g].pos) < ghost_range:
            if self.state.power_step > 0:
                self._move_defensive(g)
            else:
                self._move_aggressive(g)
        else:
            self._move_random(g)

    def _move_aggressive(self, g):
        if not self.np_random.binomial(1, p=config["_chase_prob"]):
            return self._move_random(g)

        best_dist = self.grid.x_size + self.grid.y_size
        best_pos = self.state.ghosts[g].pos
        best_dir = -1
        for d in range(self.action_space.n):
            dist = Grid.directional_distance(self.state.agent_pos,
                                             self.state.ghosts[g].pos, d)

            new_pos = self._next_pos(self.state.ghosts[g].pos, d)
            if dist <= best_dist and new_pos.is_valid() and\
                    can_move(self.state.ghosts[g], d):
                best_pos = new_pos
                best_dist = dist
                best_dir = d

        self.state.ghosts[g].update(best_pos, best_dir)

    def _move_defensive(self, g, defensive_prob=.5):
        if self.np_random.binomial(1, defensive_prob) and\
                self.state.ghosts[g].direction >= 0:
            self.state.ghosts[g].direction = -1
            return

        best_dist = 0
        best_pos = self.state.ghosts[g].pos
        best_dir = -1
        for d in range(self.action_space.n):
            dist = Grid.directional_distance(self.state.agent_pos,
                                             self.state.ghosts[g].pos, d)

            new_pos = self._next_pos(self.state.ghosts[g].pos, d)
            if dist >= best_dist and new_pos.is_valid() and\
                    can_move(self.state.ghosts[g], d):
                best_pos = new_pos
                best_dist = dist
                best_dir = d

        self.state.ghosts[g].update(best_pos, best_dir)

    def _move_random(self, g):
        # there are !!! dead ends
        # only switch to opposite direction when it failed 10 times (hack)
        ghost_pos = self.state.ghosts[g].pos
        i = 0
        while True:
            d = self.action_space.sample()
            next_pos = self._next_pos(ghost_pos, d)
            # normal map has dead ends:
            if next_pos.is_valid() and (can_move(self.state.ghosts[g], d) or
                                        i > 10):
                break
            i += 1

        self.state.ghosts[g].update(next_pos, d)


if __name__ == "__main__":
    env = PocEnv("normal")
    env.reset()
    env.render()
    done = False
    while not done:
        a = env.action_space.sample()
        ob, r, done, s = env.step(a)
        env.render()
        print("action", a, "obs", "{:010b}".format(ob), "reward", r)
    env.close()
