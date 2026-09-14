"""Trains a DQN agent on FlappyBird-v0.

Example:
    python -m flappy_bird_gymnasium.dqn.train --run-name dqn_v2 --reward legacy
"""

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from flappy_bird_gymnasium.dqn.agent import DQNAgent
from flappy_bird_gymnasium.dqn.config import DQNConfig
from flappy_bird_gymnasium.dqn.env_utils import make_env, set_global_seeds
from flappy_bird_gymnasium.rl.rewards import PRESETS

TRAIN_LOG_FIELDS = [
    "step",
    "episode",
    "return",
    "score",
    "length",
    "flap_rate",
    "epsilon",
    "loss",
    "q_mean",
]
EVAL_LOG_FIELDS = [
    "step",
    "mean_return",
    "mean_score",
    "max_score",
    "mean_length",
    "mean_flap_rate",
]


class CsvLogger:
    """Appends rows to a CSV file, writing the header on creation."""

    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    def log(self, row: Dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(row)


def evaluate(agent: DQNAgent, cfg: DQNConfig, episodes: int, seed: int) -> Dict:
    """Runs the greedy policy for `episodes` episodes on a fresh environment."""
    env = make_env(cfg, render_mode=None)
    returns, scores, lengths, flap_rates = [], [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, length, info = 0.0, 0, {"score": 0, "flaps": 0}
        while True:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            length += 1
            if terminated or truncated:
                break
        returns.append(total)
        scores.append(info["score"])
        lengths.append(length)
        flap_rates.append(info["flaps"] / max(length, 1))
    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
        "mean_length": float(np.mean(lengths)),
        "mean_flap_rate": float(np.mean(flap_rates)),
    }


def train(cfg: DQNConfig, out_dir: Path) -> Path:
    """Runs the training loop and returns the path of the best checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)

    set_global_seeds(cfg.seed)
    env = make_env(cfg, render_mode=None)
    agent = DQNAgent(env.observation_space.shape[0], int(env.action_space.n), config=cfg)
    print(
        f"Device: {agent.device} | obs_dim: {agent.obs_dim} | "
        f"reward: {cfg.reward_preset} | n_step: {cfg.n_step} | run: {out_dir}"
    )

    train_logger = CsvLogger(out_dir / "train.csv", TRAIN_LOG_FIELDS)
    eval_logger = CsvLogger(out_dir / "eval.csv", EVAL_LOG_FIELDS)

    recent_returns = deque(maxlen=50)
    recent_scores = deque(maxlen=50)
    best_score = -np.inf
    best_path = out_dir / "best.pt"
    episode = 0
    started = time.time()

    obs, _ = env.reset(seed=cfg.seed)
    ep_return, ep_length, last_metrics = 0.0, 0, None

    for step in range(1, cfg.total_steps + 1):
        epsilon = agent.epsilon(step)
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Only `terminated` ends the value bootstrap. A truncation means the
        # step limit was hit -- the episode stops, but the future was not
        # worthless, so it must not be stored as a terminal transition.
        agent.remember(obs, action, reward, next_obs, terminated)
        if truncated and not terminated:
            agent.finish_truncated_episode()

        obs = next_obs
        ep_return += reward
        ep_length += 1

        if step % cfg.train_freq == 0:
            metrics = agent.learn()
            if metrics is not None:
                last_metrics = metrics

        if terminated or truncated:
            episode += 1
            recent_returns.append(ep_return)
            recent_scores.append(info["score"])
            train_logger.log(
                {
                    "step": step,
                    "episode": episode,
                    "return": round(ep_return, 3),
                    "score": info["score"],
                    "length": ep_length,
                    "flap_rate": round(info["flaps"] / max(ep_length, 1), 4),
                    "epsilon": round(epsilon, 4),
                    "loss": round(last_metrics["loss"], 5) if last_metrics else "",
                    "q_mean": round(last_metrics["q_mean"], 3) if last_metrics else "",
                }
            )
            if episode % cfg.log_interval == 0:
                elapsed = time.time() - started
                loss_str = f"{last_metrics['loss']:.4f}" if last_metrics else "n/a"
                print(
                    f"step {step:>7} | ep {episode:>5} | "
                    f"return(50) {np.mean(recent_returns):7.2f} | "
                    f"score(50) {np.mean(recent_scores):6.2f} | "
                    f"eps {epsilon:.3f} | loss {loss_str} | "
                    f"{step / max(elapsed, 1e-9):.0f} steps/s"
                )
            obs, _ = env.reset()
            ep_return, ep_length = 0.0, 0

        if step % cfg.eval_interval == 0:
            stats = evaluate(agent, cfg, cfg.eval_episodes, seed=cfg.seed + 10_000)
            eval_logger.log({"step": step, **stats})
            print(
                f"  eval @ {step}: mean_score {stats['mean_score']:.2f} | "
                f"max_score {stats['max_score']} | "
                f"mean_return {stats['mean_return']:.2f} | "
                f"flap_rate {stats['mean_flap_rate']:.2f}"
            )
            if stats["mean_score"] > best_score:
                best_score = stats["mean_score"]
                agent.save(best_path, step=step, eval_stats=stats)
                print(f"  new best ({best_score:.2f}) -> {best_path.name}")

        if step % cfg.checkpoint_interval == 0:
            agent.save(out_dir / "latest.pt", step=step)

    agent.save(out_dir / "latest.pt", step=cfg.total_steps)
    env.close()
    print(
        f"Done in {(time.time() - started) / 60:.1f} min | "
        f"best mean score {best_score:.2f}"
    )
    return best_path if best_path.exists() else out_dir / "latest.pt"


def build_parser() -> argparse.ArgumentParser:
    defaults = DQNConfig()
    parser = argparse.ArgumentParser(description="Train a DQN agent on FlappyBird-v0.")
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--out-dir", default="runs", help="where runs are stored")
    parser.add_argument(
        "--reward",
        default=defaults.reward_preset,
        choices=sorted(PRESETS),
        help="reward scheme from flappy_bird_gymnasium.rl.rewards",
    )
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--n-step", type=int, default=defaults.n_step)
    parser.add_argument("--buffer-size", type=int, default=defaults.buffer_size)
    parser.add_argument("--learning-starts", type=int, default=defaults.learning_starts)
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=defaults.epsilon_decay_steps
    )
    parser.add_argument(
        "--target-update-interval", type=int, default=defaults.target_update_interval
    )
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument(
        "--max-episode-steps", type=int, default=defaults.max_episode_steps
    )
    parser.add_argument("--pipe-gap", type=int, default=defaults.pipe_gap)
    parser.add_argument("--no-double", action="store_true", help="plain DQN targets")
    parser.add_argument("--no-dueling", action="store_true", help="plain Q-head")
    parser.add_argument("--notes", default="", help="free-text note stored in config")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = DQNConfig(
        run_name=args.run_name,
        reward_preset=args.reward,
        total_steps=args.total_steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        n_step=args.n_step,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_interval=args.target_update_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        max_episode_steps=args.max_episode_steps,
        pipe_gap=args.pipe_gap,
        double_dqn=not args.no_double,
        dueling=not args.no_dueling,
        notes=args.notes,
    )
    train(cfg, Path(args.out_dir) / args.run_name)


if __name__ == "__main__":
    main()
