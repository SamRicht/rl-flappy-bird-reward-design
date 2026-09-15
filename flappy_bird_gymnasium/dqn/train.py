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
from flappy_bird_gymnasium.dqn.config import DQNConfig, add_config_arguments
from flappy_bird_gymnasium.dqn.env_utils import (
    limit_torch_threads,
    make_env,
    rollout,
    set_global_seeds,
    summarize_rollout,
)

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
#: Columns of eval.csv: the step plus everything `summarize_rollout` reports.
EVAL_LOG_FIELDS = [
    "step",
    "mean_return",
    "mean_score",
    "median_score",
    "std_score",
    "max_score",
    "min_score",
    "mean_length",
    "mean_flap_rate",
    "truncation_rate",
]

#: Score levels used for the sample-efficiency metric. Unlike the final score,
#: "steps until the policy sustains X" cannot be censored by the episode limit,
#: so it keeps separating variants after they all saturate.
THRESHOLDS = (1, 5, 10, 25, 50)


class CsvLogger:
    """Appends rows to a CSV file, holding the handle open for the whole run.

    Flushes after every row so a run that is killed still leaves a readable
    log, without paying for reopening the file thousands of times.
    """

    def __init__(self, path: Path, fieldnames: List[str]):
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, row: Dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def quick_eval(agent: DQNAgent, cfg: DQNConfig, episodes: int, seed: int) -> Dict:
    """Cheap greedy evaluation for the learning curve, run during training.

    Deliberately uses the *training* frame limit: a competent policy would
    otherwise make this cost more than the training itself. The price is a
    score that saturates once the policy outlives the limit, which
    `truncation_rate` reports. Numbers for a report come from `evaluate.py` or
    `summarize.py`, which measure against the higher limit.
    """
    env = make_env(cfg, render_mode=None)
    measured = rollout(env, lambda obs: agent.act(obs), episodes, seed)
    env.close()
    return summarize_rollout(measured)


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
            f"Geraet {agent.device} | Beobachtung {agent.obs_dim}-dim | "
            f"Reward {cfg.reward_preset} | n_step {cfg.n_step} | "
            f"Seed {cfg.seed} | Lauf {out_dir}"
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
    best_truncation = 0.0
    recent_evals = deque(maxlen=3)
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
                    f"Schritt {step:>7} | Episode {episode:>5} | "
                    f"Return(50) {np.mean(recent_returns):7.2f} | "
                    f"Score(50) {np.mean(recent_scores):6.2f} | "
                    f"eps {epsilon:.3f} | Loss {loss_str} | "
                    f"{step / max(elapsed, 1e-9):.0f} Schritte/s"
                )
            obs, _ = env.reset()
            ep_return, ep_length = 0.0, 0

        if step % cfg.eval_interval == 0:
            stats = quick_eval(agent, cfg, cfg.eval_episodes, seed=cfg.seed + 10_000)
            eval_logger.log({"step": step, **stats})
            recent_evals.append(stats["mean_score"])
            # Judge a checkpoint by the average of the last few evaluations,
            # not by a single one: the maximum over dozens of noisy 10-episode
            # draws is biased upwards by construction and would pick the
            # luckiest evaluation rather than the best policy.
            smoothed = float(np.mean(recent_evals))
            if verbose:
                censored = " [ZENSIERT]" if stats["truncation_rate"] >= 1.0 else ""
                print(
                    f"  Eval @ {step}: Score {stats['mean_score']:.2f}{censored} | "
                    f"geglaettet {smoothed:.2f} | max {stats['max_score']} | "
                    f"Limit-Anteil {stats['truncation_rate']:.2f} | "
                    f"Flap-Rate {stats['mean_flap_rate']:.2f}"
                )
            if smoothed > best_score:
                best_score = smoothed
                best_truncation = stats["truncation_rate"]
                agent.save(best_path, step=step, eval_stats=stats)
                if verbose:
                    print(f"  neuer Bestwert ({best_score:.2f}) -> {best_path.name}")

        if step % cfg.checkpoint_interval == 0:
            agent.save(out_dir / "latest.pt", step=step)

    agent.save(out_dir / "latest.pt", step=cfg.total_steps)
    env.close()
    train_logger.close()
    eval_logger.close()

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
        "best_eval_truncation_rate": float(best_truncation),
        "train_score_last_50": float(np.mean(recent_scores)) if recent_scores else 0.0,
        **steps_to_thresholds(episode_steps, episode_scores),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if verbose:
        print(
            f"Fertig in {minutes:.1f} min | bester Eval-Score {best_score:.2f}"
            + (
                "  (zensiert durch das Frame-Limit — echte Zahl via evaluate.py)"
                if summary["best_eval_truncation_rate"] >= 1.0
                else ""
            )
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DQN agent on FlappyBird-v0.")
    parser.add_argument("--run-name", default=DQNConfig().run_name)
    parser.add_argument("--out-dir", default="runs", help="where runs are stored")
    add_config_arguments(parser)
    # the two ablation switches read better as "turn it off" than as a value
    parser.add_argument("--no-double", action="store_true", help="plain DQN targets")
    parser.add_argument("--no-dueling", action="store_true", help="plain Q-head")
    return parser


def config_from_args(args: argparse.Namespace) -> DQNConfig:
    """Turns parsed arguments into a config.

    Every flag is named after its field, so `from_dict` does the mapping; only
    the two inverted ablation switches need a line of their own.
    """
    cfg = DQNConfig.from_dict(vars(args))
    cfg.double_dqn = not args.no_double
    cfg.dueling = not args.no_dueling
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    train(cfg, Path(args.out_dir) / cfg.run_name)


if __name__ == "__main__":
    main()
