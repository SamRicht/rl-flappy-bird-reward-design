"""Construction of the (vectorised) training and evaluation environments."""

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv

import flappy_bird_gymnasium  # noqa: F401  (registers "FlappyBird-v0")
from flappy_bird_gymnasium.rl.rewards import RewardConfig


@dataclass
class EnvConfig:
    """Settings of the environment itself, independent of the algorithm.

    Attributes:
        use_lidar: Use the 180-ray LIDAR observation instead of the 12 features.
        pipe_gap: Vertical size of the gap between the pipes; the main knob for
            the difficulty of the task.
        max_episode_steps: Step limit per episode.  A limit is required because a
            competent policy would otherwise play forever and a single episode
            could consume the whole training budget.  Hitting the limit is
            reported as a *truncation* and bootstrapped, not treated as death.
        normalize_reward: Wrap the environment in
            :class:`gymnasium.wrappers.NormalizeReward`, which divides rewards by
            a running estimate of the standard deviation of the discounted
            return.  Useful when comparing reward schemes of very different
            magnitude, since it removes the scale from the comparison.
        normalize_gamma: Discount used inside that wrapper; should match the
            agent's ``gamma``.
    """

    use_lidar: bool = False
    pipe_gap: int = 100
    max_episode_steps: int = 3_000
    normalize_reward: bool = False
    normalize_gamma: float = 0.99

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EnvConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def make_env(
    env_config: EnvConfig,
    reward_config: RewardConfig,
    seed: int,
    render_mode: Optional[str] = None,
    audio_on: bool = False,
) -> Callable[[], gym.Env]:
    """Returns a thunk that builds a single seeded environment.

    Args:
        env_config: Environment settings.
        reward_config: The reward scheme under study.
        seed: Seed for this particular environment instance.
        render_mode: Passed through to the environment; `None` for training.
        audio_on: Whether to play sounds (only sensible while rendering).

    Returns:
        A zero-argument callable as expected by
        :class:`gymnasium.vector.SyncVectorEnv`.
    """

    def thunk() -> gym.Env:
        env = gym.make(
            "FlappyBird-v0",
            use_lidar=env_config.use_lidar,
            pipe_gap=env_config.pipe_gap,
            normalize_obs=True,
            render_mode=render_mode,
            audio_on=audio_on,
            reward_config=reward_config,
        )
        env = gym.wrappers.TimeLimit(
            env, max_episode_steps=env_config.max_episode_steps
        )
        if env_config.normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=env_config.normalize_gamma)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_vector_env(
    env_config: EnvConfig,
    reward_config: RewardConfig,
    n_envs: int,
    seed: int,
) -> SyncVectorEnv:
    """Builds ``n_envs`` environments with consecutive seeds.

    Uses ``SAME_STEP`` autoreset so that a step which ends an episode already
    returns the first observation of the next one, with the true final
    observation available under ``info["final_obs"]``.  The alternative
    ``NEXT_STEP`` default of Gymnasium 1.x inserts a placeholder transition that
    would have to be masked out of every loss term.
    """
    return SyncVectorEnv(
        [make_env(env_config, reward_config, seed=seed + idx) for idx in range(n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
