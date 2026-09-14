"""Hyperparameters for the DQN agent, in one place."""

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


@dataclass
class DQNConfig:
    """All knobs of the training run.

    The defaults are tuned for the 12-feature observation mode
    (``use_lidar=False``) and reach a positive score within a few hundred
    thousand environment steps on CPU.
    """

    # --- environment ---
    use_lidar: bool = False
    normalize_obs: bool = True
    # Truncate an episode once this score is reached, so that a good agent
    # does not produce endless episodes. `None` disables the limit.
    score_limit: Optional[int] = 100

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
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    train_freq: int = 1

    # --- exploration ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 150_000

    # --- targets ---
    double_dqn: bool = True
    target_update_interval: int = 1_000

    # --- run ---
    total_steps: int = 500_000
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
