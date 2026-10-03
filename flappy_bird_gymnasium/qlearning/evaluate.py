"""Measures a trained Q-table, optionally with the game on screen.

Also the tabular agent's entry into the shared evaluation chain in
:mod:`flappy_bird_gymnasium.rl.analysis`, through :data:`RUN_LOADER`.

Example:
    python -m flappy_bird_gymnasium.qlearning.evaluate \\
        --checkpoint runs/q_adaptive/best.npz --episodes 50
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from flappy_bird_gymnasium.qlearning.agent import QLearningAgent
from flappy_bird_gymnasium.rl.analysis.runs import LoadedRun, RunLoader
from flappy_bird_gymnasium.rl.rollout import (
    EVAL_SEED,
    make_env,
    rollout,
    summarize_rollout,
)


def load_run(checkpoint: Path) -> LoadedRun:
    """A Q-table checkpoint as a greedy policy, for the shared evaluation chain.

    ``coverage`` comes from the checkpoint itself, not from the run's
    summary: ``best.npz`` is saved mid-run, and its table is the one being
    measured.
    """
    agent = QLearningAgent.load(checkpoint)
    return LoadedRun(
        config=agent.cfg,
        policy=lambda obs: agent.act(obs),
        columns={
            "discretizer": agent.cfg.discretizer,
            "n_states": agent.discretizer.n_states,
            "coverage": round(agent.coverage, 4),
            # The knobs a factorial study varies, so its result table can be
            # grouped by them directly instead of parsing them back out of
            # the directory names.
            "q_init": agent.cfg.q_init,
            "algo": agent.cfg.algo,
            "n_step": agent.cfg.n_step,
            "learning_rate_mode": agent.cfg.learning_rate_mode,
        },
    )


#: Registered as ``qlearning`` in ``rl.analysis.runs.LOADERS``.
RUN_LOADER = RunLoader(
    algorithm="qlearning",
    load=load_run,
    checkpoints=("best.npz", "latest.npz"),
    metrics=("coverage",),
)


def play(
    checkpoint: Path,
    episodes: int = 50,
    render: bool = False,
    audio_on: bool = False,
    max_episode_steps: Optional[int] = None,
    seed: int = EVAL_SEED,
) -> Dict[str, float]:
    """Plays a checkpoint greedily and returns the measured metrics.

    The environment is rebuilt from the config stored inside the checkpoint,
    so the policy is measured under the reward scheme and pipe gap it was
    trained with.

    Args:
        checkpoint: A ``.npz`` file written by :meth:`QLearningAgent.save`.
        episodes: How many episodes to play.
        render: Show the game. Throttles to 30 FPS.
        audio_on: Only meaningful together with ``render``.
        max_episode_steps: Frame limit. Defaults to the config's
            ``eval_max_episode_steps``, which is far above the training limit
            on purpose: measuring against the training limit caps every good
            policy at the same score.
        seed: Base seed of the paired evaluation.

    Returns:
        The metrics from :func:`summarize_rollout`.
    """
    agent = QLearningAgent.load(checkpoint)
    cfg = agent.cfg
    limit = (
        cfg.eval_max_episode_steps if max_episode_steps is None else max_episode_steps
    )

    env = make_env(
        cfg,
        render_mode="human" if render else None,
        audio_on=audio_on,
        max_episode_steps=limit,
    )
    measured = rollout(env, lambda obs: agent.act(obs), episodes, seed)
    env.close()

    stats = summarize_rollout(measured)
    print(
        f"Checkpoint {checkpoint} | Rasterisierung {cfg.discretizer} "
        f"({agent.discretizer.n_states} Zustaende) | Reward {cfg.reward_preset} | "
        f"Schritte trainiert {agent.train_steps}"
    )
    print(
        f"{episodes} Episoden, Limit {limit} Frames, Seeds "
        f"{seed}..{seed + episodes - 1}"
    )
    print(
        f"  Score      Mittel {stats['mean_score']:.2f}  "
        f"Median {stats['median_score']:.1f}  "
        f"Std {stats['std_score']:.2f}  "
        f"min {stats['min_score']}  max {stats['max_score']}"
    )
    print(
        f"  Laenge     Mittel {stats['mean_length']:.0f} Frames  "
        f"Flap-Rate {stats['mean_flap_rate']:.3f}"
    )
    # positiv = unterhalb der Mitte; der Betrag zeigt, wie eng die Mitte
    # gehalten wird (eine schwingende Policy mittelt sich sonst auf 0 weg)
    print(
        f"  Lage       Abstand zur Lueckenmitte "
        f"{stats['mean_gap_offset']:+.1f} px (Betrag "
        f"{stats['mean_abs_gap_offset']:.1f} px), positiv = unterhalb"
    )
    print(f"  Return     Mittel {stats['mean_return']:.2f}")
    print(f"  Abdeckung  {agent.coverage:.3f} der Tabelleneintraege besucht")
    if stats["truncation_rate"] > 0.0:
        print(
            f"  WARNUNG: {stats['truncation_rate']:.0%} der Episoden liefen ins "
            f"Frame-Limit — der Score ist nach oben zensiert."
        )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-table.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true", help="show the game")
    parser.add_argument("--audio", action="store_true", help="sound (needs --render)")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="frame limit; defaults to the checkpoint's eval_max_episode_steps",
    )
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    play(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        render=args.render,
        audio_on=args.audio,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
