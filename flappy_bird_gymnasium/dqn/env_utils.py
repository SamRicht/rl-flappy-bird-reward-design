"""Environment construction, in one place so training and evaluation agree."""

import random
from typing import Optional

import gymnasium
import numpy as np
import torch

import flappy_bird_gymnasium  # noqa: F401  (registers "FlappyBird-v0")
from flappy_bird_gymnasium.dqn.config import DQNConfig
from flappy_bird_gymnasium.rl.rewards import RewardConfig


def make_env(
    config: DQNConfig,
    render_mode: Optional[str] = None,
    audio_on: bool = False,
    max_episode_steps: Optional[int] = None,
) -> gymnasium.Env:
    """Builds the Flappy Bird environment described by `config`.

    The reward scheme comes from `flappy_bird_gymnasium.rl.rewards`, the module
    shared with the PPO work, so both agents are trained against the exact same
    reward definitions and their results stay comparable.

    Args:
        render_mode: keep this `None` while training -- in "human" mode pygame
            throttles the loop to 30 FPS, which slows training down by orders
            of magnitude.
        max_episode_steps: overrides the config's limit; pass a large value (or
            `None` for the config default) when measuring how far a trained
            agent gets.
    """
    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=audio_on,
        render_mode=render_mode,
        use_lidar=config.use_lidar,
        normalize_obs=config.normalize_obs,
        pipe_gap=config.pipe_gap,
        reward_config=RewardConfig.preset(config.reward_preset),
    )
    limit = config.max_episode_steps if max_episode_steps is None else max_episode_steps
    return gymnasium.wrappers.TimeLimit(env, max_episode_steps=limit)


def set_global_seeds(seed: int) -> None:
    """Seeds python, numpy and torch for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
