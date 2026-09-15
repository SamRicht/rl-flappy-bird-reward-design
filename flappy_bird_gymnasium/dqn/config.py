"""Hyperparameters for the DQN agent, in one place."""

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


@dataclass
class DQNConfig:
    """All knobs of the training run.

    The defaults are tuned for the 12-feature observation mode
    (``use_lidar=False``) and the ``legacy`` reward scheme.
    """

    # --- environment ---
    use_lidar: bool = False
    normalize_obs: bool = True
    pipe_gap: int = 100
    # Name of a preset in flappy_bird_gymnasium.rl.rewards.PRESETS. "legacy"
    # reproduces the upstream reward exactly and is the baseline of the study.
    reward_preset: str = "legacy"
    # Step limit per episode *while training*. Needed because a competent
    # policy would otherwise play forever and a single episode could eat the
    # whole training budget. Reaching it is a truncation and is bootstrapped,
    # never treated as death.
    max_episode_steps: int = 3_000
    # Step limit *while measuring*, deliberately far higher. The dqn_v2 run
    # showed why they must differ: at 3_000 frames every evaluation episode ran
    # into the limit and reported a score of 79, while the same policy scores
    # 304 when allowed to play on. A censored score collapses every good
    # variant onto the same number, which would make a comparison worthless.
    eval_max_episode_steps: int = 20_000

    # --- network ---
    hidden: Tuple[int, ...] = (256, 256)
    dueling: bool = True

    # --- optimization ---
    learning_rate: float = 1e-4
    batch_size: int = 64
    gamma: float = 0.99
    grad_clip: Optional[float] = 10.0
    # Huber loss instead of MSE: the "+0.1 per frame" reward makes the
    # returns (and therefore the Q-values) large, MSE would blow up.
    huber_loss: bool = True

    # --- replay ---
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    train_freq: int = 1
    # Length of the return stored per transition. n=1 is textbook DQN; larger
    # n propagates the sparse pipe reward back much faster, at the price of a
    # slightly off-policy (uncorrected) return.
    n_step: int = 3

    # --- exploration ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    # The v1 run needed ~250k steps before it cleared the first pipe reliably;
    # decaying exploration well before that starves the agent of the very
    # transitions it has to learn from.
    epsilon_decay_steps: int = 200_000

    # --- targets ---
    double_dqn: bool = True
    target_update_interval: int = 1_000

    # --- run ---
    total_steps: int = 1_000_000
    seed: int = 42
    eval_interval: int = 25_000
    # 10 episodes are too noisy to pick `best.pt` from -- a lucky run can beat
    # a genuinely better policy. For the number that goes into a report, use
    # evaluate.py with 50+ episodes.
    eval_episodes: int = 20
    log_interval: int = 10  # episodes between console log lines
    checkpoint_interval: int = 50_000

    # --- bookkeeping ---
    run_name: str = "dqn"
    notes: str = field(default="")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DQNConfig":
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        # tuples survive a JSON round-trip as lists
        cfg.hidden = tuple(cfg.hidden)
        return cfg
