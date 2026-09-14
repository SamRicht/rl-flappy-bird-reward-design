"""Environment construction, in one place so training and evaluation agree."""

import random
from typing import Optional

import gymnasium
import numpy as np
import torch

import flappy_bird_gymnasium  # noqa: F401  (registers "FlappyBird-v0")


def make_env(
    use_lidar: bool = False,
    normalize_obs: bool = True,
    render_mode: Optional[str] = None,
    audio_on: bool = False,
    score_limit: Optional[int] = None,
) -> gymnasium.Env:
    """Creates the Flappy Bird environment.

    Args:
        render_mode: keep this `None` while training -- in "human" mode
            pygame throttles the loop to 30 FPS, which slows training down by
            orders of magnitude.
        score_limit: truncates the episode once the score is reached. Without
            it a well-trained agent produces episodes that never end.
    """
    return gymnasium.make(
        "FlappyBird-v0",
        audio_on=audio_on,
        render_mode=render_mode,
        use_lidar=use_lidar,
        normalize_obs=normalize_obs,
        score_limit=score_limit,
    )


def set_global_seeds(seed: int) -> None:
    """Seeds python, numpy and torch for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
