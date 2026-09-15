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
from flappy_bird_gymnasium.dqn.env_utils import (
    limit_torch_threads,
    make_env,
    set_global_seeds,
)
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
    "truncation_rate",
]

#: Score levels used for the sample-efficiency metric. Unlike the final score,
#: "steps until the policy sustains X" cannot be censored by the episode limit,
#: so it keeps separating variants after they all saturate.
THRESHOLDS = (1, 5, 10, 25, 50)


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
    """Runs the greedy policy for `episodes` episodes on a fresh environment.

    Uses the *training* step limit, not the (much higher) measuring limit: this
    evaluation exists to draw the learning curve cheaply, and a competent
    policy would otherwise make it cost more than the training itself. The
    price is that its score saturates once the policy outlives the limit --
    `truncation_rate` reports exactly when that happens, and the number for a
    report comes from `evaluate.py` or `summarize.py` instead.
    """
    env = make_env(cfg, render_mode=None)
    returns, scores, lengths, flap_rates, truncations = [], [], [], [], []
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
        truncations.append(bool(truncated and not terminated))
    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
        "mean_length": float(np.mean(lengths)),
        "mean_flap_rate": float(np.mean(flap_rates)),
        # 1.0 means every episode hit the limit -- the score is now censored
        # and no longer distinguishes better policies from each other.
        "truncation_rate": float(np.mean(truncations)),
    }


def steps_to_thresholds(
    steps: List[int],
    scores: List[float],
    thresholds: tuple = THRESHOLDS,
    window: int = 20,
) -> Dict[str, Optional[int]]:
    """Environment steps until the rolling mean score first reaches each level.

    This is the metric that survives the episode limit: it measures how fast a
    variant learned, which stays meaningful even after every variant saturates
    at the capped score. The rolling mean keeps one lucky episode from counting
    as "reached".
    """
    result: Dict[str, Optional[int]] = {f"steps_to_{t}": None for t in thresholds}
    if len(scores) < window:
        return result

    rolling = np.convolve(np.asarray(scores, dtype=float), np.ones(window) / window, "valid")
    # rolling[i] covers episodes i .. i+window-1, credited to the last of them
    rolling_steps = np.asarray(steps[window - 1 :])
    for threshold in thresholds:
        hit = np.flatnonzero(rolling >= threshold)
        if hit.size:
            result[f"steps_to_{threshold}"] = int(rolling_steps[hit[0]])
    return result


def train(
    cfg: DQNConfig,
    out_dir: Path,
    verbose: bool = True,
    torch_threads: Optional[int] = None,
) -> Dict:
    """Runs the training loop and returns a summary of the run.

    Args:
        verbose: print progress lines. Turn off for runs inside a pool, where
            interleaved output from several runs is unreadable anyway.
        torch_threads: pin Torch to this many threads. Pass 1 when several runs
            share a machine, otherwise they oversubscribe the CPU and each one
            ends up slower than it would be alone.
    """
    if torch_threads is not None:
        limit_torch_threads(torch_threads)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)

    set_global_seeds(cfg.seed)
    env = make_env(cfg, render_mode=None)
    agent = DQNAgent(env.observation_space.shape[0], int(env.action_space.n), config=cfg)
    if verbose:
        print(
            f"Device: {agent.device} | obs_dim: {agent.obs_dim} | "
            f"reward: {cfg.reward_preset} | n_step: {cfg.n_step} | "
            f"seed: {cfg.seed} | run: {out_dir}"
        )

    train_logger = CsvLogger(out_dir / "train.csv", TRAIN_LOG_FIELDS)
    eval_logger = CsvLogger(out_dir / "eval.csv", EVAL_LOG_FIELDS)

    recent_returns = deque(maxlen=50)
    recent_scores = deque(maxlen=50)
    # kept for the censoring-free sample-efficiency metric at the end
    episode_steps: List[int] = []
    episode_scores: List[float] = []
    best_score = -np.inf
    best_path = out_dir / "best.pt"
    best_stats: Dict = {}
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
            episode_steps.append(step)
            episode_scores.append(float(info["score"]))
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
            if verbose and episode % cfg.log_interval == 0:
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
            if verbose:
                censored = " [ZENSIERT]" if stats["truncation_rate"] >= 1.0 else ""
                print(
                    f"  eval @ {step}: mean_score {stats['mean_score']:.2f}{censored} | "
                    f"max_score {stats['max_score']} | "
                    f"trunc {stats['truncation_rate']:.2f} | "
                    f"flap_rate {stats['mean_flap_rate']:.2f}"
                )
            if stats["mean_score"] > best_score:
                best_score = stats["mean_score"]
                best_stats = stats
                agent.save(best_path, step=step, eval_stats=stats)
                if verbose:
                    print(f"  new best ({best_score:.2f}) -> {best_path.name}")

        if step % cfg.checkpoint_interval == 0:
            agent.save(out_dir / "latest.pt", step=step)

    agent.save(out_dir / "latest.pt", step=cfg.total_steps)
    env.close()

    minutes = (time.time() - started) / 60
    summary = {
        "run_name": cfg.run_name,
        "run_dir": str(out_dir),
        "seed": cfg.seed,
        "reward_preset": cfg.reward_preset,
        "total_steps": cfg.total_steps,
        "episodes": episode,
        "minutes": round(minutes, 2),
        "best_eval_score": float(best_score),
        "best_eval_truncation_rate": float(best_stats.get("truncation_rate", 0.0)),
        "train_score_last_50": float(np.mean(recent_scores)) if recent_scores else 0.0,
        **steps_to_thresholds(episode_steps, episode_scores),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if verbose:
        print(
            f"Done in {minutes:.1f} min | best eval score {best_score:.2f}"
            + (
                "  (zensiert durch das Schrittlimit -- echte Zahl via evaluate.py)"
                if summary["best_eval_truncation_rate"] >= 1.0
                else ""
            )
        )
    return summary


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
        "--max-episode-steps",
        type=int,
        default=defaults.max_episode_steps,
        help="frame limit while training (keeps episodes affordable)",
    )
    parser.add_argument(
        "--eval-max-episode-steps",
        type=int,
        default=defaults.eval_max_episode_steps,
        help="frame limit while measuring; must exceed what the policy reaches",
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
        eval_max_episode_steps=args.eval_max_episode_steps,
        pipe_gap=args.pipe_gap,
        double_dqn=not args.no_double,
        dueling=not args.no_dueling,
        notes=args.notes,
    )
    train(cfg, Path(args.out_dir) / args.run_name)


if __name__ == "__main__":
    main()
