"""How the shared evaluation chain reads the runs of the PPO work.

The chain in :mod:`flappy_bird_gymnasium.rl.analysis` knows nothing about any
learning method.  It needs two things from each of them: the layout of a run
directory, which the PPO training already writes, and a
:class:`~flappy_bird_gymnasium.rl.analysis.runs.RunLoader` turning a checkpoint
into a greedy policy.  This module is that loader.

Docking on rather than keeping our own evaluation is the point: a number from
the PPO work and a number from the tabular or DQN work are only comparable if
they were produced by the same code, on the same episode seeds, under the same
frame limit.  Our own ``rl/summarize.py`` measures the same quantities but not
provably the same way.

One thing does not survive the move.  Our evaluation pools the episodes of five
late checkpoints, because a single one measures one arbitrary moment of a
heavy-tailed process.  The chain measures one checkpoint per run, so a fair
comparison across methods and our own most stable estimate are, for now, two
separate measurements.  ``model_late_00.pt`` .. ``model_late_04.pt`` are listed
below so that each can at least be measured individually through the chain.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from flappy_bird_gymnasium.rl.analysis.runs import LoadedRun, RunLoader
from flappy_bird_gymnasium.rl.ppo import ActorCritic, PPOConfig

#: Frame limit for the final measurement, far above the training limit.  At 36
#: frames per pipe this caps the score near 25,000; the training limit of 3,000
#: frames would cap every decent policy at the same ~80 and hide the whole
#: difference between the reward designs.
EVAL_MAX_EPISODE_STEPS = 900_000

#: Checkpoints a PPO run holds.  ``model.pt`` is first and therefore the
#: default: it is the only one whose selection does not depend on a measurement.
#: ``model_best.pt`` was picked by its own rolling score and reading it as "the
#: result" would report the best of some 700 noisy windows.
CHECKPOINTS: Tuple[str, ...] = (
    "model.pt",
    "model_best.pt",
    "model_late_00.pt",
    "model_late_01.pt",
    "model_late_02.pt",
    "model_late_03.pt",
    "model_late_04.pt",
)


@dataclass(frozen=True)
class PPORunConfig:
    """A PPO run's configuration in the shape the chain expects.

    The chain reads its configs duck-typed, so this carries exactly the
    attributes :mod:`flappy_bird_gymnasium.rl.rollout` and the summarising stage
    look at -- no more, and under the chain's names rather than ours.

    ``normalize_obs`` has no counterpart in ``EnvConfig``: the PPO training
    passes it as a constant ``True`` when building the environment, so it is one
    here as well.  ``reward_overrides`` carries every term of the scheme, not
    just the ones that differ from the preset, which reproduces the reward
    exactly even if a preset is edited later.
    """

    seed: int
    gamma: float
    use_lidar: bool
    pipe_gap: int
    max_episode_steps: int
    reward_preset: str
    reward_overrides: Dict[str, object] = field(default_factory=dict)
    normalize_obs: bool = True
    eval_max_episode_steps: int = EVAL_MAX_EPISODE_STEPS


def _config_from_run(run_dir: Path) -> PPORunConfig:
    """Translates a run's ``config.json`` into :class:`PPORunConfig`."""
    stored = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    reward = dict(stored["reward"])
    name = str(reward.pop("name"))
    return PPORunConfig(
        seed=int(stored["ppo"]["seed"]),
        gamma=float(stored["ppo"]["gamma"]),
        use_lidar=bool(stored["env"]["use_lidar"]),
        pipe_gap=int(stored["env"]["pipe_gap"]),
        max_episode_steps=int(stored["env"]["max_episode_steps"]),
        reward_preset=name,
        reward_overrides=reward,
    )


def _greedy_policy(model: ActorCritic):
    """Wraps a trained network into ``obs -> action``, always taking the arg max.

    Greedy on purpose.  Sampling is how PPO explores while it learns; measuring
    a sampling policy would mix what the agent learned with how much randomness
    it still carries, and the two move in opposite directions over training.
    """

    def policy(obs: np.ndarray) -> int:
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
            features = model.backbone(tensor)
            logits = model.actor_head(model.actor_body(features))
            return int(torch.argmax(logits, dim=-1).item())

    return policy


def load(checkpoint: Path) -> LoadedRun:
    """Builds the policy stored in ``checkpoint``.

    Module-level rather than a closure or a method because the chain sends it
    to worker processes, which pickle it by name.

    Args:
        checkpoint: Path to a checkpoint file inside a run directory.

    Returns:
        The run, ready to be measured.
    """
    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent
    config = _config_from_run(run_dir)

    stored = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ppo_config = PPOConfig.from_dict(stored["ppo"])
    model = ActorCritic(
        180 if config.use_lidar else 12,
        2,
        hidden_sizes=ppo_config.hidden_sizes,
        shared_backbone=ppo_config.shared_backbone,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    return LoadedRun(config=config, policy=_greedy_policy(model))


#: Registered in ``analysis.runs.LOADERS`` as ``"ppo"``.
RUN_LOADER = RunLoader(
    algorithm="ppo",
    load=load,
    checkpoints=CHECKPOINTS,
    # The PPO training predates the shared contract and writes its per-episode
    # log under its own name, with the environment step in `global_step`.
    train_log="episodes.csv",
    step_column="global_step",
)
