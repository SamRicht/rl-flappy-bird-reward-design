"""Environment construction, in one place so training and evaluation agree."""

import random
from typing import Callable, Dict, Optional

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
) -> gymnasium.Env:
    """Builds the Flappy Bird environment described by `config`.

    The reward scheme comes from `flappy_bird_gymnasium.rl.rewards`, the module
    shared with the PPO work, so both agents are trained against the exact same
    reward definitions and their results stay comparable.

    Args:
        render_mode: keep this `None` while training -- in "human" mode pygame
            throttles the loop to 30 FPS, which slows training down by orders
            of magnitude.
        max_episode_steps: frame limit per episode. Defaults to the training
            limit; pass `config.eval_max_episode_steps` when measuring, because
            measuring against the training limit censors the score of every
            competent policy at the same value.
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


def rollout(
    env: gymnasium.Env,
    policy: Callable[[np.ndarray], int],
    episodes: int,
    seed: int,
) -> Dict[str, list]:
    """Plays `episodes` episodes and returns the per-episode measurements.

    Every measurement in this package goes through here — the trained agents
    and the random reference alike — so the reference is by construction
    measured exactly like the thing it is a reference for.

    Args:
        policy: anything that maps an observation to an action, e.g.
            `lambda obs: agent.act(obs)` or `lambda obs: rng.integers(2)`.
        seed: episodes use `seed, seed + 1, …`. Two policies measured with the
            same seed see identical pipe layouts, which makes the comparison
            paired instead of a matter of who drew the easier levels.
    """
    measured: Dict[str, list] = {
        "return": [],
        "score": [],
        "length": [],
        "flap_rate": [],
        "truncated": [],
    }
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total, length, info = 0.0, 0, {"score": 0, "flaps": 0}
        while True:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            length += 1
            if terminated or truncated:
                break
        measured["return"].append(total)
        measured["score"].append(info["score"])
        measured["length"].append(length)
        measured["flap_rate"].append(info["flaps"] / max(length, 1))
        # only a truncation that is not also a crash means "cut short"
        measured["truncated"].append(bool(truncated and not terminated))
    return measured


def summarize_rollout(measured: Dict[str, list]) -> Dict[str, float]:
    """Condenses a `rollout` result into the metrics every caller reports."""
    return {
        "mean_return": float(np.mean(measured["return"])),
        "mean_score": float(np.mean(measured["score"])),
        "median_score": float(np.median(measured["score"])),
        "std_score": float(np.std(measured["score"])),
        "max_score": int(np.max(measured["score"])),
        "min_score": int(np.min(measured["score"])),
        "mean_length": float(np.mean(measured["length"])),
        "mean_flap_rate": float(np.mean(measured["flap_rate"])),
        # 1.0 means every episode hit the frame limit: the score is censored
        # from above and no longer tells better policies apart
        "truncation_rate": float(np.mean(measured["truncated"])),
    }


def set_global_seeds(seed: int) -> None:
    """Seeds python, numpy and torch for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
