"""Hyperparameters of the Q-learning agent, in one place."""

import argparse
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

from flappy_bird_gymnasium.qlearning.discretize import DISCRETIZERS
from flappy_bird_gymnasium.rl.rewards import PRESETS

#: Update rules the agent implements. ``expected_sarsa`` replaces the max in
#: the target with the expectation under the current epsilon-greedy policy.
#: That needs no committed next action, so the training loop stays the same.
#: On-policy SARSA would need one, which is why it's not here.
ALGOS: Tuple[str, ...] = ("qlearning", "expected_sarsa")

#: How the step size is chosen, see ``learning_rate_mode``.
LEARNING_RATE_MODES: Tuple[str, ...] = ("linear", "count")


@dataclass
class QLearningConfig:
    """All knobs of a training run.

    The environment fields have the same names as in the DQN work because
    :func:`flappy_bird_gymnasium.rl.rollout.make_env` reads them by name.
    Same names also keeps the runs comparable: same ``pipe_gap`` and same
    reward definition means both agents were given the same task.
    """

    # --- environment ---
    # the 12-feature observation. The env default is use_lidar=True (180
    # dims), which can't be binned into a usable table.
    use_lidar: bool = False
    normalize_obs: bool = True
    pipe_gap: int = 100
    # preset name from rl.rewards.PRESETS. "legacy" is the upstream reward
    # and the baseline of the study.
    reward_preset: str = "legacy"
    # single reward terms overridden on top of the preset, e.g.
    # {"ceiling": 0.0}. The presets differ in several terms at once; this is
    # how one term of one scheme gets isolated.
    reward_overrides: Dict[str, float] = field(default_factory=dict)
    # step limit per episode while training. Without it a decent policy
    # plays forever and one episode could eat the whole budget. Hitting it
    # is a truncation (bootstrapped), never a death.
    max_episode_steps: int = 3_000
    # step limit while measuring, much higher on purpose. Measuring against
    # the training limit caps every good policy at the same score and
    # collapses the variants onto one number. truncation_rate shows when it
    # starts to bite.
    eval_max_episode_steps: int = 50_000
    # step limit of the periodic eval during training. Above the training
    # limit so the learning curve (and the best.npz selection) doesn't
    # saturate, below the final one to stay cheap.
    eval_quick_max_episode_steps: int = 10_000

    # --- rasterisation ---
    # key in flappy_bird_gymnasium.qlearning.discretize.DISCRETIZERS
    discretizer: str = "uniform"
    # bin counts of the uniform scheme. Too coarse merges fatal decisions
    # into one cell, too fine starves the table of visits.
    dx_bins: int = 24
    dy_bins: int = 32
    # vel_y takes exactly 20 integer values, so 20 bins resolve it without
    # aliasing and more would just leave gaps
    vel_bins: int = 20

    # --- learning ---
    learning_rate: float = 0.3
    # annealed linearly to this; equal to learning_rate means constant. A
    # constant step size never lets the table settle (stochastic env, the
    # values keep bouncing).
    learning_rate_end: float = 0.02
    # None means over the whole run, so the annealing follows the budget
    # instead of quietly finishing after a tenth of it
    learning_rate_decay_steps: Optional[int] = None
    # "linear": one annealed step size for the whole table.
    # "count": one per entry, alpha / (1 + visits)**exponent, floored at
    # learning_rate_end. Matters at the end of a run: with one global
    # schedule the often-visited and the rarely-visited entries settle at
    # the same time, although the latter have seen a fraction of the data.
    learning_rate_mode: str = "linear"
    # Robbins-Monro wants an exponent in (0.5, 1]; 0.6 is the usual pick
    learning_rate_exponent: float = 0.6
    # length of the return backed up per update. 1 is textbook Q-learning;
    # larger values carry the sparse +1 of a passed pipe back over more
    # states at once, at the cost of a slightly off-policy return.
    n_step: int = 1
    gamma: float = 0.99
    # initial value of every table entry, the single most important knob
    # here. Above the achievable return every untried action looks good, so
    # the agent tries it once (optimistic initialisation). With 0.0 the
    # greedy policy locks onto whichever action first got a positive value
    # and never tries the other.
    #
    # 10.0 is optimistic for every preset in rl.rewards: the densest scheme
    # ("legacy", +0.1 per frame) has a value of 0.1/(1-gamma) = 10 for
    # surviving forever, the sparse ones considerably less. A scheme with
    # larger rewards needs a larger value here, see the README.
    q_init: float = 10.0
    algo: str = "qlearning"

    # --- exploration ---
    # q_init already explores systematically, so epsilon only needs a short
    # decay. In Flappy Bird a random action at the wrong moment is fatal.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 300_000

    # --- run ---
    # far more than the DQN budget: a tabular step has no forward and no
    # backward pass, and a table needs many visits per state instead of one
    # generalising update. --total-steps 2000000 for a quick look.
    total_steps: int = 10_000_000
    seed: int = 42
    # scaled with the budget: each periodic eval plays up to
    # eval_episodes * eval_quick_max_episode_steps frames, which at a shorter
    # interval would cost more than the training itself
    eval_interval: int = 500_000
    # 15 episodes are noisy; the number for a report comes from evaluate.py
    # with 50+ episodes at the higher frame limit. The checkpoint is picked
    # on the median of these, not the mean: the score distribution has a
    # long right tail.
    eval_episodes: int = 15
    log_interval: int = 50  # episodes between console log lines
    checkpoint_interval: int = 1_000_000

    # --- bookkeeping ---
    run_name: str = "qlearning"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}, got {self.algo!r}")
        if self.learning_rate_mode not in LEARNING_RATE_MODES:
            raise ValueError(
                f"learning_rate_mode must be one of {LEARNING_RATE_MODES}, "
                f"got {self.learning_rate_mode!r}"
            )
        if self.n_step < 1:
            raise ValueError(f"n_step must be at least 1, got {self.n_step}")
        if self.discretizer not in DISCRETIZERS:
            raise ValueError(
                f"unknown discretizer {self.discretizer!r}; "
                f"available: {sorted(DISCRETIZERS)}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "QLearningConfig":
        """Builds a config from any dict, ignoring keys that aren't fields.

        Extra keys are tolerated so that ``from_dict(vars(args))`` works
        without a hand-written argument-to-field mapping.
        """
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds one CLI flag per tunable field, each named after its field.

    Because ``--learning-rate`` lands in ``args.learning_rate``,
    ``QLearningConfig.from_dict(vars(args))`` builds the config without a
    mapping to keep in sync. Every entry point calls this, so their flags
    can't drift apart.
    """
    defaults = QLearningConfig()
    add = parser.add_argument

    add("--total-steps", type=int, default=defaults.total_steps)
    add("--seed", type=int, default=defaults.seed)
    add(
        "--reward-preset",
        "--reward",
        dest="reward_preset",
        default=defaults.reward_preset,
        choices=sorted(PRESETS),
        help="reward scheme from flappy_bird_gymnasium.rl.rewards",
    )

    add(
        "--discretizer",
        default=defaults.discretizer,
        choices=sorted(DISCRETIZERS),
        help="state rasterisation scheme",
    )
    add(
        "--dx-bins",
        type=int,
        default=defaults.dx_bins,
        help="bins for the horizontal distance ('uniform' only)",
    )
    add(
        "--dy-bins",
        type=int,
        default=defaults.dy_bins,
        help="bins for the vertical gap offset ('uniform' only)",
    )
    add(
        "--vel-bins",
        type=int,
        default=defaults.vel_bins,
        help="bins for the vertical velocity ('uniform' only)",
    )

    add("--learning-rate", type=float, default=defaults.learning_rate)
    add(
        "--learning-rate-end",
        type=float,
        default=defaults.learning_rate_end,
        help="step size after annealing; equal to --learning-rate means constant",
    )
    add(
        "--learning-rate-decay-steps",
        type=int,
        default=defaults.learning_rate_decay_steps,
    )
    add(
        "--learning-rate-mode",
        default=defaults.learning_rate_mode,
        choices=list(LEARNING_RATE_MODES),
        help="one annealed step size for the table, or one per entry by visits",
    )
    add(
        "--learning-rate-exponent",
        type=float,
        default=defaults.learning_rate_exponent,
        help="exponent of the per-entry step size ('count' mode only)",
    )
    add(
        "--n-step",
        type=int,
        default=defaults.n_step,
        help="length of the return backed up per update",
    )
    add("--gamma", type=float, default=defaults.gamma)
    add(
        "--q-init",
        type=float,
        default=defaults.q_init,
        help="initial table value; above the achievable return it explores by itself",
    )
    add("--algo", default=defaults.algo, choices=list(ALGOS))

    add("--epsilon-start", type=float, default=defaults.epsilon_start)
    add("--epsilon-end", type=float, default=defaults.epsilon_end)
    add("--epsilon-decay-steps", type=int, default=defaults.epsilon_decay_steps)

    add("--pipe-gap", type=int, default=defaults.pipe_gap)
    add(
        "--max-episode-steps",
        type=int,
        default=defaults.max_episode_steps,
        help="frame limit while training (keeps episodes affordable)",
    )
    add(
        "--eval-max-episode-steps",
        type=int,
        default=defaults.eval_max_episode_steps,
        help="frame limit while measuring; must exceed what the policy reaches",
    )
    add(
        "--eval-quick-max-episode-steps",
        type=int,
        default=defaults.eval_quick_max_episode_steps,
        help="frame limit of the periodic evaluation during training",
    )
    add("--eval-interval", type=int, default=defaults.eval_interval)
    add("--eval-episodes", type=int, default=defaults.eval_episodes)
    add(
        "--log-interval",
        type=int,
        default=defaults.log_interval,
        help="episodes between console log lines",
    )
    add("--notes", default=defaults.notes, help="free-text note stored in config.json")
    return parser
