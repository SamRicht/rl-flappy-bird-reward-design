"""Environment construction, in one place so training and evaluation agree."""

import random
from typing import Optional

import gymnasium
import numpy as np
import torch

import flappy_bird_gymnasium  # noqa: F401  (registers "FlappyBird-v0")
from flappy_bird_gymnasium.dqn.config import DQNConfig
from flappy_bird_gymnasium.rl.rewards import RewardConfig


def limit_torch_threads(n_threads: int = 1) -> None:
    """Pins Torch to `n_threads`, so a pool of runs does not oversubscribe the CPU.

    A single DQN run on this observation size gains nothing from intra-op
    parallelism, but several runs each spawning as many threads as there are
    cores will fight over them and each end up slower than if it ran alone.
    """
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(n_threads)
    except RuntimeError:
        # already initialised -- harmless, the pool worker is fresh anyway
        pass


def make_env(
    config: DQNConfig,
    render_mode: Optional[str] = None,
    audio_on: bool = False,
    max_episode_steps: Optional[int] = None,
    evaluation: bool = False,
) -> gymnasium.Env:
    """Builds the Flappy Bird environment described by `config`.

    The reward scheme comes from `flappy_bird_gymnasium.rl.rewards`, the module
    shared with the PPO work, so both agents are trained against the exact same
    reward definitions and their results stay comparable.

    Args:
        render_mode: keep this `None` while training -- in "human" mode pygame
            throttles the loop to 30 FPS, which slows training down by orders
            of magnitude.
        evaluation: use the (much higher) measuring limit instead of the
            training limit. Measuring against the training limit censors the
            score of every competent policy at the same value.
        max_episode_steps: overrides both limits explicitly.
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
    if max_episode_steps is None:
        max_episode_steps = (
            config.eval_max_episode_steps if evaluation else config.max_episode_steps
        )
    return gymnasium.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)


def set_global_seeds(seed: int) -> None:
    """Seeds python, numpy and torch for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
