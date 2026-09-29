"""Environment construction and measurement, shared by every learning method.

All agents in the project should train against the same environment definition
and be measured with the same procedure. Two numbers are only comparable if
they were produced the same way, and the easiest way to make sure of that is
to have exactly one implementation.

No deep-learning dependency in here: a tabular agent needs NumPy and nothing
else, and the CI workflow doesn't install PyTorch. Methods that use a framework
seed it themselves, on top of :func:`set_global_seeds`.

The ``config`` objects passed around here are duck-typed. Only these
attributes are read::

    use_lidar  normalize_obs  pipe_gap  reward_preset  reward_overrides
    gamma      max_episode_steps

Any dataclass with those names works, so one implementation can serve several
agents without a common base class.
"""

import math
import random
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

import gymnasium
import numpy as np

import flappy_bird_gymnasium  # noqa: F401  (registers "FlappyBird-v0")
from flappy_bird_gymnasium.rl.geometry import env_gap_offset
from flappy_bird_gymnasium.rl.rewards import RewardConfig

#: Base seed of every final measurement. Episode ``i`` runs with ``seed + i``,
#: so two policies measured with the same base seed see the same pipe layouts
#: and the comparison is paired, also across methods. Far from the small
#: integers used as training seeds, so the two can't be confused.
EVAL_SEED = 10_000


def reward_config_for(config) -> RewardConfig:
    """The reward scheme of ``config``: its preset, overrides applied, gamma tied.

    ``reward_overrides`` replaces single terms of the preset. That's how one
    term of a scheme gets isolated: whole presets differ in several terms at
    once, so a difference between two of them can't be pinned on any one of
    them.

    Potential-based shaping only leaves the optimal policy unchanged when its
    discount equals the agent's. The preset default (0.99) happens to match
    the usual agent default, but a gamma sweep would silently break the
    guarantee, so both are set from the same value.

    Args:
        config: Anything with ``reward_preset``, ``reward_overrides`` and
            ``gamma``.

    Returns:
        The reward scheme to hand to the environment.
    """
    reward = RewardConfig.preset(config.reward_preset, **config.reward_overrides)
    if reward.uses_shaping:
        reward = replace(reward, shaping_gamma=config.gamma)
    return reward


def make_env(
    config,
    render_mode: Optional[str] = None,
    audio_on: bool = False,
    max_episode_steps: Optional[int] = None,
    wrapper: Optional[Callable[[gymnasium.Env], gymnasium.Env]] = None,
    **env_kwargs: Any,
) -> gymnasium.Env:
    """Builds the Flappy Bird environment described by ``config``.

    ``FlappyBird-v0`` is registered without a ``max_episode_steps``, so
    ``gymnasium.make`` adds no ``TimeLimit`` on its own. Without the wrapper
    added here the first decent policy would play forever and a single
    episode could eat the whole training budget.

    A method that sees the game differently, e.g. the CNN work through
    stacked pixel frames, passes its observation wrappers as ``wrapper`` and
    anything else ``FlappyBird-v0`` takes (``background="night"``) as keyword
    arguments. Reward scheme and frame limit then still come from here, so
    they can't drift from what every other method uses.

    Args:
        config: See the module docstring for the attributes that are read.
        render_mode: Keep this ``None`` while training. In "human" mode
            pygame throttles the loop to 30 FPS, which slows training down by
            orders of magnitude.
        audio_on: Only meaningful together with ``render_mode="human"``.
        max_episode_steps: Frame limit per episode. Defaults to the training
            limit from the config; pass ``config.eval_max_episode_steps``
            when measuring, since the training limit caps every good policy
            at the same score.
        wrapper: Applied to the bare environment, inside the frame limit.
        **env_kwargs: Further arguments for ``FlappyBird-v0``.

    Returns:
        The wrapped environment.
    """
    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=audio_on,
        render_mode=render_mode,
        use_lidar=config.use_lidar,
        normalize_obs=config.normalize_obs,
        pipe_gap=config.pipe_gap,
        reward_config=reward_config_for(config),
        **env_kwargs,
    )
    if wrapper is not None:
        env = wrapper(env)
    limit = config.max_episode_steps if max_episode_steps is None else max_episode_steps
    return gymnasium.wrappers.TimeLimit(env, max_episode_steps=limit)


def rollout(
    env: gymnasium.Env,
    policy: Callable[[np.ndarray], int],
    episodes: int,
    seed: int,
) -> Dict[str, List]:
    """Plays ``episodes`` episodes and returns the per-episode measurements.

    Every measurement in the project should go through here, the trained
    agents and the random reference alike, so the reference is measured
    exactly like the thing it's a reference for.

    Args:
        env: An environment built by :func:`make_env`.
        policy: Anything mapping an observation to an action, e.g.
            ``lambda obs: agent.act(obs)`` or ``lambda obs: rng.integers(2)``.
        episodes: How many episodes to play.
        seed: Episodes use ``seed, seed + 1, ...``. Two policies measured
            with the same seed see identical pipe layouts, which makes the
            comparison paired.

    Returns:
        One list per metric, one entry per episode. The two gap offsets are
        read off the game state, not the observation, so they exist under
        every observation type; ``nan`` only for an episode that ended before
        a pipe came on screen.
    """
    measured: Dict[str, List] = {
        "return": [],
        "score": [],
        "length": [],
        "flap_rate": [],
        "truncated": [],
        "gap_offset": [],
        "abs_gap_offset": [],
    }
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total, length, info = 0.0, 0, {"score": 0, "flaps": 0}
        offsets: List[float] = []
        while True:
            # measured in the state the policy is about to act on, so it
            # describes where the bird flew and not where it ended up
            offset = env_gap_offset(env)
            if offset is not None:
                offsets.append(offset)
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            length += 1
            if terminated or truncated:
                break
        measured["return"].append(total)
        measured["score"].append(info["score"])
        measured["length"].append(length)
        measured["flap_rate"].append(info["flaps"] / max(length, 1))
        # only a truncation that isn't also a crash counts as "cut short"
        measured["truncated"].append(bool(truncated and not terminated))
        measured["gap_offset"].append(_mean_or_nan(offsets))
        measured["abs_gap_offset"].append(_mean_or_nan([abs(o) for o in offsets]))
    return measured


def _mean_or_nan(values: List[float]) -> float:
    """Mean of ``values``, or ``nan`` for an empty list.

    ``nan`` and not 0.0 on purpose: a zero offset means "flew dead centre",
    which is the best possible value, so 0.0 would quietly turn a missing
    measurement into an excellent one.
    """
    return float(np.mean(values)) if values else float("nan")


def summarize_rollout(measured: Dict[str, List]) -> Dict[str, float]:
    """Condenses a :func:`rollout` result into the metrics every caller reports.

    The keys are the contract between the methods: they're the columns of
    ``eval.csv`` and of the aggregated result tables, so a row from one agent
    can sit next to a row from another.
    """
    return {
        "mean_return": float(np.mean(measured["return"])),
        "mean_score": float(np.mean(measured["score"])),
        "median_score": float(np.median(measured["score"])),
        "std_score": float(np.std(measured["score"])),
        "max_score": int(np.max(measured["score"])),
        "min_score": int(np.min(measured["score"])),
        "mean_length": float(np.mean(measured["length"])),
        "mean_flap_rate": float(np.mean(measured["flap_rate"])),
        # 1.0 means every episode hit the frame limit; the score is then
        # capped and no longer tells better policies apart
        "truncation_rate": float(np.mean(measured["truncated"])),
        # behaviour, not performance: two reward schemes can reach the same
        # score while one hugs the gap centre and the other skims its edge.
        # Signed shows a systematic bias (flies high or low), absolute shows
        # how tightly the centre is held. A policy that oscillates
        # symmetrically averages to zero without ever being centred.
        "mean_gap_offset": _mean_or_nan(
            [v for v in measured["gap_offset"] if not math.isnan(v)]
        ),
        "mean_abs_gap_offset": _mean_or_nan(
            [v for v in measured["abs_gap_offset"] if not math.isnan(v)]
        ),
    }


def set_global_seeds(seed: int) -> None:
    """Seeds the standard library and NumPy for a reproducible run.

    Agents that bring their own framework seed it on top of this. Agents with
    a private generator (the usual choice, it keeps their behaviour
    independent of whatever else draws from the global streams) only need
    this for the environment.
    """
    random.seed(seed)
    np.random.seed(seed)
