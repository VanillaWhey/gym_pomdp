import os
from itertools import zip_longest

import pygame
import numpy as np
from gym_pomdp.envs.coord import Coord, Tile

PATH = os.path.split(__file__)[0]
FILE_PATH = os.path.join(PATH, 'assets')


class GuiTile(Tile):
    _borderWidth = 2
    _borderColor = pygame.Color("grey")

    def __init__(self, coord: Coord, surface, tile_size=100):
        self.origin = (coord.x * tile_size, coord.y * tile_size)
        self.surface = surface
        self.tile_size = tuple([tile_size] * 2)
        self.value = None
        super().__init__(coord)

    def draw(self, img=None, color=pygame.Color("white")):
        rect = pygame.Rect(self.origin, self.tile_size)
        pygame.draw.rect(self.surface, color, rect, 0)  # draw tile
        if img is not None:
            self.surface.blit(img, self.origin)
        pygame.draw.rect(self.surface, GuiTile._borderColor, rect,
                         GuiTile._borderWidth)  # draw border

    def set_value(self, value):
        self.value = value


class GridGui(object):
    _assets = {}

    def __init__(self, x_size, y_size, tile_size):
        self.x_size = x_size
        self.y_size = y_size
        self.n_tiles = x_size * y_size
        self.tile_size = tile_size
        self.assets = {}
        for k, v in self._assets.items():
            self.assets[k] = pygame.transform.scale(pygame.image.load(v),
                                                    [tile_size] * 2)

        self.w = self.tile_size * self.x_size
        self.h = (self.tile_size * self.y_size) + 50  # size of the taskbar

        pygame.init()
        self.surface = pygame.display.set_mode((self.w, self.h))
        self.surface.fill(pygame.Color("white"))
        self.action_font = pygame.font.SysFont("monospace", 18)
        self.board = []
        self.build_board()

    def build_board(self):
        self.board = []
        for idx in range(self.n_tiles):
            tile = GuiTile(self.get_coord(idx), surface=self.surface,
                           tile_size=self.tile_size)
            tile.draw(img=None)
            self.board.append(tile)

    def get_coord(self, idx):
        assert 0 <= idx < self.n_tiles
        return Coord(idx % self.x_size, idx // self.x_size)

    def draw(self, update_board=False):
        raise NotImplementedError()

    def render(self, state, msg):
        raise NotImplementedError()

    def task_bar(self, msg):
        assert msg is not None
        txt = self.action_font.render(msg, 2, pygame.Color("black"))
        rect = pygame.Rect((0, self.h - 50 + 5), (self.w, 40))  # 205 for tiger
        pygame.draw.rect(self.surface, pygame.Color("white"), rect, 0)
        self.surface.blit(txt, (self.tile_size // 2, self.h - 50 + 10))  # 210

    @staticmethod
    def _dispatch():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None


class ShipGui(GridGui):
    _tile_size = 50

    def __init__(self, board_size=(5, 5), obj_pos=()):
        super().__init__(*board_size, tile_size=self._tile_size)

        self.obj_pos = obj_pos
        self.last_shot = -1
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def draw(self, update_board=False):
        if update_board:
            for pos in self.obj_pos:
                self.board[pos].draw(color=pygame.Color("black"))
        else:
            if self.last_shot in self.obj_pos:
                self.board[self.last_shot].draw(color=pygame.Color("red"))
            elif self.last_shot > -1:
                self.board[self.last_shot].draw(color=pygame.Color("blue"))

    def render(self, state, msg):
        self.last_shot = state
        self.draw()
        self.task_bar(msg)
        pygame.display.update()
        GridGui._dispatch()


class RockGui(GridGui):
    _tile_size = 50
    _assets = dict(
        _ROBOT=os.path.join(FILE_PATH, "r2d2.png"),
        _ROCK=os.path.join(FILE_PATH, "rock.png")
    )

    def __init__(self, board_size=(5, 5), start_pos=0, obj=()):
        super().__init__(*board_size, tile_size=self._tile_size)
        self.history = [start_pos] * 2
        self.last_pos = start_pos
        self.pos = start_pos
        self.obj = np.array(obj)
        self.offset = self.x_size * (self.y_size - 1)
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def draw(self, update_board=False):

        if update_board:
            for obj in self.obj:
                if obj[1] == -1:
                    color = pygame.Color('red')
                elif obj[1] == 1:
                    color = pygame.Color('blue')
                else:
                    color = pygame.Color('grey')
                self.board[obj[0]].draw(img=self.assets["_ROCK"], color=color)

        last_state = self.history.pop(0)
        idx = np.where(self.obj[:, 0] == last_state)[0]
        if idx.size > 0:
            if self.obj[idx[0], 1] == -1:
                color = pygame.Color('red')
            else:
                color = pygame.Color('blue')
            self.board[last_state].draw(img=self.assets["_ROCK"], color=color)
        else:
            self.board[last_state].draw()

        self.board[self.history[-1]].draw(img=self.assets["_ROBOT"])

    def render(self, state, msg=None):
        self.history.append(state)
        self.draw(update_board=True)
        if msg is not None:
            self.task_bar(msg)
        pygame.display.update()
        GridGui._dispatch()


class TagGui(GridGui):
    _tile_size = 50
    _assets = dict(
        _ROBOT=os.path.join(FILE_PATH, "r2d2.png"),
        _STORM=os.path.join(FILE_PATH, "soldier.png"),
        _RIGHT=os.path.join(FILE_PATH, "right.png"),
        _LEFT=os.path.join(FILE_PATH, "left.png"),
        _UP=os.path.join(FILE_PATH, "up.png"),
        _DOWN=os.path.join(FILE_PATH, "down.png"),

    )

    def __init__(self, board_size=(10, 5), start_pos=0, obj_pos=()):
        super().__init__(*board_size, tile_size=self._tile_size)
        self.obj_pos = obj_pos
        self.agent_history = [start_pos] * 2
        self.opp_history = []
        self.update_opp(obj_pos)
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def draw(self, update_board=False):
        if update_board:
            for t in self.board:
                t.draw()

        for opp in self.opp_history:
            old, new = opp
            self.board[old].draw()
            if new is not None:
                self.board[new].draw(img=self.assets["_STORM"])

        self.opp_history = []

        last_state = self.agent_history.pop(0)
        self.board[last_state].draw()
        self.board[self.agent_history[-1]].draw(img=self.assets["_ROBOT"])

    def render(self, state, msg=None):

        agent_pos, obj_pos = state
        self.agent_history.append(agent_pos)
        self.update_opp(obj_pos)
        self.draw(update_board=True)
        if msg is not None:
            self.task_bar(msg)

        pygame.display.update()
        GridGui._dispatch()

    def update_opp(self, opp_pos):
        self.opp_history = []
        for old, new in zip_longest(self.obj_pos, opp_pos):
            self.opp_history.append((old, new))
        self.obj_pos = opp_pos


class TigerGui(GridGui):
    _tile_size = 200
    _assets = dict(
        _CAT=os.path.join(PATH, "assets/cat.png"),
        _DOOR=os.path.join(PATH, "assets/door.png"),
        _BEER=os.path.join(PATH, "assets/beer.png"))

    def __init__(self, board_size=(2, 1)):
        super(TigerGui, self).__init__(*board_size, tile_size=self._tile_size)

        self.draw(update_board=True)
        pygame.display.update()
        TigerGui._dispatch()

    def draw(self, update_board=False):
        if update_board:
            for tile in self.board:
                tile.draw(img=self.assets["_DOOR"])

    def render(self, state, msg=None):

        if msg is not None:
            self.task_bar(msg)

        action, state = state

        if action < 2:
            if action == state:
                self.board[state].draw(img=self.assets["_BEER"])
            else:
                self.board[action].draw(img=self.assets["_CAT"])
        else:
            self.draw(update_board=True)

        pygame.display.update()
        TigerGui._dispatch()


class PocGui(GridGui):
    _tile_size = 50
    _assets = dict(
        _POWER=os.path.join(PATH, "assets/power.png"),
        _FOOD=os.path.join(PATH, "assets/food.png"),
        _POC_0=os.path.join(PATH, "assets/poc_r.png"),
        _POC_1=os.path.join(PATH, "assets/poc_d.png"),
        _POC_2=os.path.join(PATH, "assets/poc_l.png"),
        _POC_3=os.path.join(PATH, "assets/poc_u.png"),
        _GHOST_0=os.path.join(PATH, "assets/ghost1.png"),
        _GHOST_1=os.path.join(PATH, "assets/ghost2.png"),
        _GHOST_2=os.path.join(PATH, "assets/ghost3.png"),
        _GHOST_3=os.path.join(PATH, "assets/ghost4.png"),
        _GHOST_P=os.path.join(PATH, "assets/ghost_p.png")
    )

    def __init__(self, state, board_size=(5, 5), maze=None):
        super().__init__(board_size[1], board_size[0],
                         tile_size=self._tile_size)
        self.state = state
        self.maze = maze
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def init_board(self, maze):
        m = np.ravel(maze)
        for idx in range(self.n_tiles):
            if m[idx] == 4:
                self.board[idx].draw(img=self.assets["_FOOD"],
                                     color=pygame.Color("black"))
            elif m[idx] == 0:
                self.board[idx].draw(color=pygame.Color("blue"))
            elif m[idx] == 3:
                self.board[idx].draw(img=self.assets["_POWER"],
                                     color=pygame.Color("black"))
            else:
                self.board[idx].draw(color=pygame.Color("black"))

    def draw(self, update_board=False):
        # reset board
        self.init_board(self.maze)
        # draw PocMan
        self.board[self.__idx(self.state.agent_pos)].draw(
            img=self.assets["_POC_" + str(self.state.action)],
            color=pygame.Color("black"))
        # draw ghosts
        g = 0
        for ghost in [self.__idx(ghost.pos) for ghost in self.state.ghosts]:
            if self.state.power_step > 0:
                ghost_img = self.assets["_GHOST_P"]
            else:
                ghost_img = self.assets["_GHOST_" + str(g)]
            self.board[ghost].draw(img=ghost_img, color=pygame.Color("black"))
            g += 1

    def render(self, state, msg=None):
        self.state = state
        self.maze[self.state.agent_pos] = 1
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def __idx(self, pos):
        return pos[0] * self.x_size + pos[1]


def calc_tile_size(length):
    if length >= 70:
        return 25
    if length >= 50:
        return 30
    return 50


class TMazeGui(GridGui):
    _assets = dict(_ROBOT=os.path.join(FILE_PATH, "r2d2.png"))

    def __init__(self, state, board_size=(5, 3), goal=-1):
        super().__init__(*board_size, tile_size=calc_tile_size(board_size[0]))
        self.state = state
        self.last = state
        self. goal = (2 + goal) * self.x_size - 1
        self.init_board()
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def init_board(self):
        for idx in range(self.n_tiles):
            if self.x_size <= idx < self.x_size * 2 or\
                    (idx + 1) % self.x_size == 0:

                self.board[idx].draw(color=pygame.Color("white"))
            else:
                self.board[idx].draw(color=pygame.Color("black"))
            if idx == self.goal:
                self.board[idx].draw(color=pygame.Color("yellow"))

    def draw(self, update_board=False):
        # draw robot
        self.board[self.x_size + self.last].draw(color=pygame.Color("white"))
        if self.state >= 0:
            self.board[self.x_size + self.state].draw(
                img=self.assets["_ROBOT"])
        else:
            idx = (-self.state) * self.x_size - 1
            if idx == self.goal:
                color = pygame.Color("yellow")
            else:
                color = pygame.Color("white")

            self.board[idx].draw(img=self.assets["_ROBOT"], color=color)

    def render(self, state, msg=None):
        self.last = self.state
        self.state = state
        self.draw(update_board=True)
        self.task_bar(msg)
        pygame.display.update()
        GridGui._dispatch()
