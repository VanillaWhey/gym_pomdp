from enum import IntEnum, unique

import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding

from gym_pomdp.envs.coord import Coord, Grid, Moves
from gym_pomdp.envs.gui import RockGui


@unique
class Obs(IntEnum):
    NULL = 0
    GOOD = 2
    BAD = 1


@unique
class Action(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    SAMPLE = 4


def action_to_str(action):
    if action == Action.NORTH:
        return "north"
    elif action == Action.EAST:
        return "east"
    elif action == Action.SOUTH:
        return "south"
    elif action == Action.WEST:
        return "west"
    elif action == Action.SAMPLE:
        return "sample"
    elif action > Action.SAMPLE:
        return "check"
    else:
        raise NotImplementedError()


config = {
    # 2, 1
    2: {"init_pos": (0, 0),
        "rock_pos": [[1, 0]]},
    # 4, 3
    4: {"init_pos": (0, 0),
        "rock_pos": [[1, 0], [3, 1], [2, 3]]},
    # 7, 8
    7: {"init_pos": (0, 3),
        "rock_pos": [[2, 0], [0, 1], [3, 1], [6, 3],
                     [2, 4], [3, 4], [5, 5], [1, 6]]},
    # 11, 11
    11: {"init_pos": (0, 5),
         "rock_pos": [[0, 3], [0, 7], [1, 8], [2, 4], [3, 3], [3, 8],
                      [4, 3], [5, 8], [6, 1], [9, 3], [9, 9]],
         },
    # 15, 15
    15: {"init_pos": (0, 7),
         "rock_pos":
             [[0, 7], [0, 3], [1, 2], [2, 6], [3, 7],
              [3, 2], [4, 7], [5, 2], [6, 9], [9, 7],
              [9, 1], [11, 8], [12, 2], [13, 10], [14, 9]]},
}


class Rock(object):
    def __init__(self, pos, status):
        self.status = status
        self.pos = pos
        self.count = 0
        self.measured = 0
        self.lkw = 1.  # likely worthless
        self.lkv = 1.  # likely valuable
        self.prob_valuable = .5


class RockState(object):
    def __init__(self, pos):
        self.agent_pos = pos
        self.rocks = []


class RockEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, board_size=7, num_rocks=8, use_heuristic=False,
                 observation='o', stay_inside=False, p_move=1.0):
        """

        :param board_size: int board is a square of board_size x board_size
        :param num_rocks: int number of rocks on board
        :param use_heuristic: bool usage unclear
        :param observation: str must be one of
                                'o': observed value only
                                'po': position of the agent + the above
                                'poa': the above + the action taken
        """

        assert board_size in list(config.keys()) and \
               num_rocks == len(config[board_size]["rock_pos"])

        self.np_random = None
        self.seed()

        self.num_rocks = num_rocks
        self._use_heuristic = use_heuristic

        self._rock_pos = \
            [Coord(*rock) for rock in config[board_size]['rock_pos']]
        self._agent_pos = Coord(*config[board_size]['init_pos'])
        self.grid = Grid(board_size, board_size)

        for idx, rock in enumerate(self._rock_pos):
            self.grid.board[rock] = idx

        self.p_move = p_move
        if p_move < 1.0:
            self._penalization = 0
        else:
            self._penalization = -100

        self.action_space = Discrete(len(Action) + self.num_rocks)
        self._discount = .95
        self._reward_range = 20
        self._query = 0
        if stay_inside:
            self._out_of_bounds_penalty = 0
        else:
            self._out_of_bounds_penalty = self._penalization

        self.state = None
        self.last_action = None
        self.done = False

        self.gui = None

        assert observation in ['o', 'oa', 'po', 'poa']
        if observation == 'o':
            self._make_obs = lambda obs, a: obs
            self.observation_space = Discrete(len(Obs))
        elif observation == 'oa':
            self._make_obs = self._oa
            self.observation_space =\
                Box(low=0,
                    high=np.append(max(Obs), np.ones(self.action_space.n)),
                    dtype=np.int)

        elif observation == 'po':
            self._make_obs = self._po
            self.observation_space = \
                Box(low=0,
                    high=np.append(np.ones(self.grid.n_tiles), max(Obs)),
                    dtype=np.int)

        elif observation == 'poa':
            self._make_obs = self._poa
            self.observation_space = \
                Box(low=0,
                    high=np.concatenate((np.ones(self.grid.n_tiles),
                                         [max(Obs)],
                                        np.ones(self.action_space.n))),
                    dtype=np.int)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        assert self.done is False

        self.last_action = action
        self._query += 1

        reward = 0
        ob = Obs.NULL

        if self.np_random.binomial(1, p=self.p_move):
            if action < Action.SAMPLE:
                if action == Action.EAST:
                    if self.state.agent_pos.x + 1 < self.grid.x_size:
                        self.state.agent_pos += Moves.EAST.value
                    else:
                        reward = 10
                        self.done = True
                        ob = self._make_obs(ob.value, action)
                        info = {"state": self._encode_state(self.state)}
                        return ob, reward, self.done, info
                elif action == Action.NORTH:
                    if self.state.agent_pos.y + 1 < self.grid.y_size:
                        self.state.agent_pos += Moves.NORTH.value
                    else:
                        reward = self._out_of_bounds_penalty
                elif action == Action.SOUTH:
                    if self.state.agent_pos.y - 1 >= 0:
                        self.state.agent_pos += Moves.SOUTH.value
                    else:
                        reward = self._out_of_bounds_penalty
                elif action == Action.WEST:
                    if self.state.agent_pos.x - 1 >= 0:
                        self.state.agent_pos += Moves.WEST.value
                    else:
                        reward = self._out_of_bounds_penalty
                else:
                    raise NotImplementedError()

            elif action == Action.SAMPLE:
                rock = self.grid[self.state.agent_pos]
                # collected
                if rock >= 0 and not self.state.rocks[rock].status == 0:
                    if self.state.rocks[rock].status == 1:
                        reward = 10
                    else:
                        reward = -10
                    self.state.rocks[rock].status = 0
                else:
                    reward = self._penalization

            elif action > Action.SAMPLE:
                rock = action - Action.SAMPLE - 1
                assert rock < self.num_rocks

                eff = RockEnv._efficiency(self.state.agent_pos,
                                          self.state.rocks[rock].pos)

                if self.np_random.binomial(1, eff):
                    ob = Obs(1 + self.state.rocks[rock].status)
                else:
                    ob = Obs(2 - self.state.rocks[rock].status)

                self.state.rocks[rock].measured += 1

                if ob == Obs.GOOD:
                    self.state.rocks[rock].count += 1
                    self.state.rocks[rock].lkv *= eff
                    self.state.rocks[rock].lkw *= (1 - eff)
                else:
                    self.state.rocks[rock].count -= 1
                    self.state.rocks[rock].lkw *= eff
                    self.state.rocks[rock].lkv *= (1 - eff)

                    denominator = (.5 * self.state.rocks[rock].lkv) + \
                                  (.5 * self.state.rocks[rock].lkw) + 1e-10
                    self.state.rocks[rock].prob_valuable = \
                        (.5 * self.state.rocks[rock].lkv) / denominator

        self.done = self._penalization == reward
        ob = self._make_obs(ob.value, action)
        return ob, reward, self.done, {"state": self._encode_state(self.state)}

    def _decode_state(self, state, as_array=False):

        agent_pos = Coord(*state['agent_pos'])
        rock_state = RockState(agent_pos)
        for r in state['rocks']:
            rock = Rock(pos=0, status=self.np_random.choice(2))
            rock.__dict__.update(r)
            rock_state.rocks.append(rock)

        if as_array:
            rocks = []
            for rock in rock_state.rocks:
                rocks.append(rock.status)

            return np.concatenate([[self.grid.get_index(agent_pos)], rocks])

        return rock_state

    @staticmethod
    def _encode_state(state):
        # use dictionary for state encoding
        enc_state = {}
        for k, v in vars(state).items():
            if isinstance(v, list):
                ll = []
                for idx, t in enumerate(v):
                    ll.append(vars(t))
                v = ll
            enc_state[k] = v
        return enc_state

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == "human":
            msg = None
            if self.gui is None:
                start_pos = self.grid.get_index(self.state.agent_pos)
                obj_pos = [(self.grid.get_index(rock.pos), rock.status) for
                           rock in self.state.rocks]
                self.gui = RockGui((self.grid.x_size, self.grid.y_size),
                                   start_pos=start_pos, obj=obj_pos)

            if self.last_action > Action.SAMPLE:
                rock = self.last_action - Action.SAMPLE - 1
                msg = "Rock S: {} P:{}".format(self.state.rocks[rock].status,
                                               self.state.rocks[rock].pos)
            agent_pos = self.grid.get_index(self.state.agent_pos)
            self.gui.render(agent_pos, msg)

    def reset(self):
        self.done = False
        self._query = 0
        self.last_action = Action.SAMPLE
        self.state = self._get_init_state(should_encode=False)
        return self._make_obs(Obs.NULL, self.last_action)

    def _set_state(self, state):
        self.done = False
        self.state = self._decode_state(state)

    def close(self):
        self.render(close=True)

    def _compute_prob(self, action, next_state, ob):

        next_state = self._decode_state(next_state)

        if action <= Action.SAMPLE:
            return int(ob == Obs.NULL)

        eff = self._efficiency(next_state.agent_pos, next_state.rocks[
            action - Action.SAMPLE - 1].pos)

        if ob == Obs.GOOD and next_state.rocks[
           action - Action.SAMPLE - 1].status == 1:
            return eff
        elif ob == Obs.BAD and next_state.rocks[
             action - Action.SAMPLE - 1].status == -1:
            return eff
        else:
            return 1 - eff

    def _get_init_state(self, should_encode=True):

        rock_state = RockState(self._agent_pos)
        for idx in range(self.num_rocks):
            rock_state.rocks.append(Rock(pos=self._rock_pos[idx],
                                         status=self.np_random.choice(2)))
        return self._encode_state(rock_state) if should_encode else rock_state

    def _generate_legal(self):
        legal = [Action.EAST]  # can always go east
        if self.state.agent_pos.y + 1 < self.grid.y_size:
            legal.append(Action.NORTH)

        if self.state.agent_pos.y - 1 >= 0:
            legal.append(Action.SOUTH)
        if self.state.agent_pos.x - 1 >= 0:
            legal.append(Action.WEST)

        rock = self.grid[self.state.agent_pos]
        if rock >= 0 and self.state.rocks[rock].status != 0:
            legal.append(Action.SAMPLE)

        for rock in self.state.rocks:
            assert self.grid[rock.pos] != -1
            if rock.status != 0:
                legal.append(self.grid[rock.pos] + 1 + Action.SAMPLE)
        return legal

    def _generate_preferred(self, history):
        if not self._use_heuristic:
            return self._generate_legal()

        actions = []

        # sample rocks with high likelihood of being good
        rock = self.grid[self.state.agent_pos]
        if rock >= 0 and self.state.rocks[rock].status != 0 and history.size:
            total = 0
            # history
            for t in range(history.size):
                if history[t].action == rock + 1 + Action.SAMPLE:
                    if history[t].ob == Obs.GOOD:
                        total += 1
                    elif history[t].ob == Obs.BAD:
                        total -= 1
            if total > 0:
                actions.append(Action.SAMPLE)
                return actions

        # process the rocks

        all_bad = True
        direction = {
            "north": False,
            "south": False,
            "west": False,
            "east": False
        }
        for idx in range(self.num_rocks):
            rock = self.state.rocks[idx]
            if rock.status != 0:
                total = 0
                for t in range(history.size):
                    if history[t].action == idx + 1 + Action.SAMPLE:
                        if history[t].ob == Obs.GOOD:
                            total += 1
                        elif history[t].ob == Obs.BAD:
                            total -= 1
                if total >= 0:
                    all_bad = False

                    if rock.pos.y > self.state.agent_pos.y:
                        direction['north'] = True
                    elif rock.pos.y < self.state.agent_pos.y:
                        direction['south'] = True
                    elif rock.pos.x < self.state.agent_pos.x:
                        direction['west'] = True
                    elif rock.pos.x > self.state.agent_pos.x:
                        direction['east'] = True

        if all_bad:
            actions.append(Action.EAST)
            return actions

        # generate a random legal move
        # do not measure a collected rock
        # do no measure a rock too often
        # do not measure clearly bad rocks
        # don't move in a direction that puts you closer to bad rocks
        # never sample a rock

        if self.state.agent_pos.y + 1 < self.grid.y_size and\
                direction['north']:
            actions.append(Action.NORTH)

        if direction['east']:
            actions.append(Action.EAST)

        if self.state.agent_pos.y - 1 >= 0 and direction['south']:
            actions.append(Action.SOUTH)

        if self.state.agent_pos.x - 1 >= 0 and direction['west']:
            actions.append(Action.WEST)

        for idx, rock in enumerate(self.state.rocks):
            if not rock.status == 0 and rock.measured < 5 and abs(
                    rock.count) < 2 and 0 < rock.prob_valuable < 1:
                actions.append(idx + 1 + Action.SAMPLE)

        if len(actions) == 0:
            return self._generate_legal()

        return actions

    def __dict2np__(self, state):
        idx = self.grid.get_index(Coord(*state['agent_pos']))
        rocks = []
        for rock in state['rocks']:
            rocks.append(rock['status'])
        return np.concatenate([[idx], rocks])

    @staticmethod
    def _efficiency(agent_pos, rock_pos, hed=20):
        d = Grid.euclidean_distance(agent_pos, rock_pos)
        eff = (1 + pow(2, -d / hed)) * .5
        return eff

    @staticmethod
    def _select_target(rock_state, x_size):
        best_dist = x_size * 2
        best_rock = -1  # Coord(-1, -1)
        for idx, rock in enumerate(rock_state.rocks):
            if rock.status != 0 and rock.count >= 0:
                d = Grid.manhattan_distance(rock_state.agent_pos, rock.pos)
                if d < best_dist:
                    best_dist = d
                    best_rock = idx  # rock.pos
        return best_rock

    def _po(self, o, _):
        obs = np.zeros(self.observation_space.shape[0])
        obs[self.grid.x_size * self.state.agent_pos.y +
            self.state.agent_pos.x] = 1.
        obs[self.grid.n_tiles] = o
        return obs

    def _poa(self, o, a):
        obs = self._po(o, a)
        obs[self.grid.n_tiles + a] = 1.
        return obs

    def _oa(self, o, a):
        obs = np.zeros(self.observation_space.shape[0])
        obs[0] = o
        obs[1 + a] = 1.
        return obs
