"""Tabular Q-learning on FlappyBird-v0.

Unlike the network-based agents this one keeps a single value per state-action
pair. That makes the rasterisation of the continuous observation into a finite
state space a hyperparameter of its own, and it is the one knob no other method
in the project has.

Entry points, each runnable as a module::

    python -m flappy_bird_gymnasium.qlearning.train --run-name q1
    python -m flappy_bird_gymnasium.qlearning.evaluate --checkpoint runs/q1/best.npz
"""

from flappy_bird_gymnasium.qlearning.agent import QLearningAgent
from flappy_bird_gymnasium.qlearning.config import ALGOS, QLearningConfig
from flappy_bird_gymnasium.qlearning.discretize import (
    DISCRETIZERS,
    Discretizer,
    build_discretizer,
    pipes_ahead,
)

__all__ = [
    "ALGOS",
    "DISCRETIZERS",
    "Discretizer",
    "QLearningAgent",
    "QLearningConfig",
    "build_discretizer",
    "pipes_ahead",
]
