"""Evaluates a trained DQN checkpoint, optionally with the game window open.

Examples:
    python -m flappy_bird_gymnasium.dqn.evaluate --checkpoint runs/dqn/best.pt
    python -m flappy_bird_gymnasium.dqn.evaluate --checkpoint runs/dqn/best.pt --render
"""

import argparse
from pathlib import Path
from typing import List, Optional

from flappy_bird_gymnasium.dqn.agent import DQNAgent
from flappy_bird_gymnasium.dqn.env_utils import make_env, rollout, summarize_rollout


def play(
    checkpoint: Path,
    episodes: int = 10,
    render: bool = False,
    audio: bool = False,
    max_episode_steps: Optional[int] = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Measures a checkpoint's greedy policy and reports the result.

    The environment is rebuilt from the config stored in the checkpoint, so the
    agent is always measured under the reward scheme it was trained on — and by
    default against the measuring frame limit, not the training one. See the
    README section on censored scores for why that distinction matters.
    """
    agent = DQNAgent.load(checkpoint)
    agent.q_net.eval()

    limit = max_episode_steps or agent.cfg.eval_max_episode_steps
    env = make_env(
        agent.cfg,
        render_mode="human" if render else None,
        audio_on=audio and render,
        max_episode_steps=limit,
    )
    if verbose:
        print(
            f"Reward-Preset {agent.cfg.reward_preset} | n_step {agent.cfg.n_step} | "
            f"Trainings-Seed {agent.cfg.seed} | Frame-Limit {limit}"
        )

    measured = rollout(env, lambda obs: agent.act(obs), episodes, seed)
    env.close()

    if verbose:
        for index in range(episodes):
            print(
                f"Episode {index + 1:>3}: Score {measured['score'][index]:>4} | "
                f"Return {measured['return'][index]:8.2f} | "
                f"Frames {measured['length'][index]:>5} | "
                f"Flap-Rate {measured['flap_rate'][index]:.2f}"
                + ("  (Limit erreicht)" if measured["truncated"][index] else "")
            )

    summary = {"episodes": episodes, "frame_limit": int(limit)}
    summary.update(summarize_rollout(measured))

    if verbose:
        print(
            f"\n{episodes} Episoden | Score {summary['mean_score']:.2f} "
            f"+/- {summary['std_score']:.2f} "
            f"(Median {summary['median_score']:.0f}, "
            f"min {summary['min_score']}, max {summary['max_score']}) | "
            f"Flap-Rate {summary['mean_flap_rate']:.2f}"
        )
        if summary["truncation_rate"] > 0:
            print(
                f"WARNUNG: {summary['truncation_rate']:.0%} der Episoden liefen ins "
                f"Frame-Limit von {limit}. Der Score ist nach oben zensiert — "
                f"mit hoeherem --max-episode-steps nachmessen."
            )
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="show the game window")
    parser.add_argument("--audio", action="store_true", help="sound (needs --render)")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="frame limit per episode (default: the run's measuring limit)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    play(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        render=args.render,
        audio=args.audio,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
