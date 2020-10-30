## gym-pomdp
This package is an extensions of OpenAI Gym, for Partially Observable Markov Decision Process.

## Dependencies
- Python
- OpenAI Gym
- PyGame

## Installation
Check out the latest code
```git clone https://github.com/VanillaWhey/gym_pomdp.git```

To use it as a package simply run:
```bash
cd gym_pomdp
pip install -e .
```

## Usage
Import the library and gym as and call the environment:
```python
import gym
import gym_pomdp
env = gym.make("Tag-v0")
```

## Implemented envs
All environments are parametrized as in the original papers. In order to get larger state space or more enemies, it's easy to change the board_size
in the specific environment.

- Tag
- Tiger
- BattleShip
- Network
- RockSample
- Pocman
- T-Maze

## Recommended readings
[General overview](http://cs.mcgill.ca/~jpineau/talks/jpineau-dagstuhl13.pdf)
[POMCP solver](https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf)
[Point-based value iteration](http://www.fore.robot.cc/papers/Pineau03a.pdf)
[Similar work](https://github.com/pemami4911/POMDPy)
[RL-LSTM](http://papers.nips.cc/paper/1953-reinforcement-learning-with-long-short-term-memory.pdf)

## Special thanks
David Silver and Joel Veness made this possible by releasing the code POMCP open source. 
And [@manuel](https://github.com/manuel-delverme) for proof test.

## TODO
- correct rendered orientation of Pocman [minor issue]
