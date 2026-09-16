"""Reinforcement learning code for the Flappy Bird environment.

Kept free of top-level imports of :mod:`flappy_bird_gymnasium` itself so that
``flappy_bird_gymnasium.envs`` can import :mod:`flappy_bird_gymnasium.rl.rewards`
during package initialisation without a circular import.
"""
