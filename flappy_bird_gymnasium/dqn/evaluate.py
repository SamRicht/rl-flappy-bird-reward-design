"""Evaluates a trained DQN checkpoint, optionally with the game window open.

Examples:
    python -m flappy_bird_gymnasium.dqn.evaluate --checkpoint runs/dqn/best.pt
    python -m flappy_bird_gymnasium.dqn.evaluate --checkpoint runs/dqn/best.pt --render
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from flappy_bird_gymnasium.dqn.agent import DQNAgent
from flappy_bird_gymnasium.dqn.env_utils import make_env


def play(
    checkpoint: Path,
    episodes: int = 10,
    render: bool = False,
    audio: bool = False,
    score_limit: Optional[int] = None,
    seed: int = 0,
    epsilon: float = 0.0,
) -> dict:
    """Runs the (near-)greedy policy and prints per-episode results."""
    agent = DQNAgent.load(checkpoint)
    agent.q_net.eval()

    env = make_env(
        use_lidar=agent.cfg.use_lidar,
        normalize_obs=agent.cfg.normalize_obs,
        render_mode="human" if render else None,
        audio_on=audio and render,
        score_limit=score_limit,
    )

    returns, scores, lengths = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, length, info = 0.0, 0, {"score": 0}
        while True:
            action = agent.act(obs, epsilon=epsilon)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            length += 1
            if terminated or truncated:
                break
        returns.append(total)
        scores.append(info["score"])
        lengths.append(length)
        print(
            f"episode {i + 1:>3}: score {info['score']:>4} | "
            f"return {total:8.2f} | frames {length:>5}"
        )
    env.close()

    summary = {
        "episodes": episodes,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
    }
    print(
        f"\n{episodes} episodes | mean score {summary['mean_score']:.2f} "
        f"(median {summary['median_score']:.0f}, "
        f"min {summary['min_score']}, max {summary['max_score']}) | "
        f"mean return {summary['mean_return']:.2f}"
    )
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="show the game window")
    parser.add_argument("--audio", action="store_true", help="sound (needs --render)")
    parser.add_argument(
        "--score-limit",
        type=int,
        default=None,
        help="stop an episode at this score (default: no limit)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="exploration during evaluation, 0 = fully greedy",
    )
    args = parser.parse_args(argv)
    play(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        render=args.render,
        audio=args.audio,
        score_limit=args.score_limit,
        seed=args.seed,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    main()
