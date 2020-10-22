import logging

from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="Tiger-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:TigerEnv"
)
register(
    id="Pocman-v0",
    # max_episode_steps = 400,
    kwargs={"maze":"micro"},
    entry_point="gym_pomdp.envs:PocEnv"
)
register(
    id="Pocman-v1",
    # max_episode_steps = 400,
    kwargs={"maze":"mini"},
    entry_point="gym_pomdp.envs:PocEnv"
)
register(
    id="Pocman-v2",
    # max_episode_steps = 400,
    kwargs={"maze":"normal"},
    entry_point="gym_pomdp.envs:PocEnv"
)
register(
    id="Tag-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:TagEnv"
)
# Battleship
register(
    id="Battleship-v1",
    # max_episode_steps = 400,
    kwargs={"board_size": (10, 10), "ships": [2, 2, 3, 4, 5]},
    entry_point="gym_pomdp.envs:BattleShipEnv"
)
register(
    id="BattleshipSmall-v0",
    # max_episode_steps = 400,
    kwargs={"board_size": (5, 5), "ships": [2, 3, 4]},
    entry_point="gym_pomdp.envs:BattleShipEnv"
)
register(
    id="Battleship-v0",
    # max_episode_steps = 400,
    kwargs={"board_size": (5, 5), "ships": [2, 3, 4]},
    entry_point="gym_pomdp.envs:BattleShipEnv"
)
# RockSample
register(
    id="RockSample-v0",
    # max_episode_steps = 400,
    kwargs={"board_size": 7, "num_rocks": 8},
    entry_point="gym_pomdp.envs:RockEnv"
)
register(
    id="RockSample-v1",
    # max_episode_steps = 400,
    kwargs={"board_size": 11, "num_rocks": 11},
    entry_point="gym_pomdp.envs:RockEnv"
)
register(
    id="RockSample-v2",
    # max_episode_steps = 400,
    kwargs={"board_size": 15, "num_rocks": 15},
    entry_point="gym_pomdp.envs:RockEnv"
)
register(
    id="RockSample-v3",
    # max_episode_steps = 400,
    kwargs={"board_size": 2, "num_rocks": 2},
    entry_point="gym_pomdp.envs:RockEnv"
)
register(
    id="StochasticRock-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:StochasticRockEvn"
)
register(
    id="Network-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:NetworkEnv"
)
register(
    id="Test-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:TestEnv"
)
# t-maze
register(
    id="TMaze-v0",
    # max_episode_steps = 400,
    kwargs={"length": 5},
    entry_point="gym_pomdp.envs:TMazeEnv"
)
register(
    id="TMaze-v1",
    # max_episode_steps = 400,
    kwargs={"length": 30},
    entry_point="gym_pomdp.envs:TMazeEnv"
)
register(
    id="TMaze-v2",
    # max_episode_steps = 400,
    kwargs={"length": 50},
    entry_point="gym_pomdp.envs:TMazeEnv"
)
register(
    id="TMaze-v3",
    # max_episode_steps = 400,
    kwargs={"length": 70},
    entry_point="gym_pomdp.envs:TMazeEnv"
)
