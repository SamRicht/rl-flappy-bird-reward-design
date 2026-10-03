"""Trains a tabular Q-learning agent on FlappyBird-v0.

Example:
    python -m flappy_bird_gymnasium.qlearning.train --run-name q1 --reward legacy
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from flappy_bird_gymnasium.qlearning.agent import QLearningAgent
from flappy_bird_gymnasium.qlearning.config import QLearningConfig, add_config_arguments
from flappy_bird_gymnasium.qlearning.discretize import build_discretizer
from flappy_bird_gymnasium.rl.rollout import (
    make_env,
    rollout,
    set_global_seeds,
    summarize_rollout,
)
from flappy_bird_gymnasium.rl.runlog import (
    EVAL_LOG_FIELDS,
    CsvLogger,
    steps_to_thresholds,
)

#: Columns of ``train.csv``. The first seven are shared with the other methods,
#: ``alpha``, ``td_error`` and ``coverage`` are tabular-specific diagnostics.
TRAIN_LOG_FIELDS = [
    "step",
    "episode",
    "return",
    "score",
    "length",
    "flap_rate",
    "epsilon",
    "alpha",
    "td_error",
    "coverage",
]


def quick_eval(
    agent: QLearningAgent, cfg: QLearningConfig, episodes: int, seed: int
) -> Dict:
    """Greedy evaluation for the learning curve, run during training.

    Uses ``eval_quick_max_episode_steps`` instead of the training limit, so
    good policies aren't all capped at the same score. Not the number for a
    report; that one comes from ``evaluate.py`` with the higher limit and
    more episodes.
    """
    env = make_env(cfg, max_episode_steps=cfg.eval_quick_max_episode_steps)
    measured = rollout(env, lambda obs: agent.act(obs), episodes, seed)
    env.close()
    return summarize_rollout(measured)


def train(cfg: QLearningConfig, out_dir: Path, verbose: bool = True) -> Dict:
    """Runs the training loop and returns a summary of the run.

    Args:
        cfg: The hyperparameters.
        out_dir: Directory for ``config.json``, the CSV logs, the checkpoints
            and ``summary.json``. Created if missing.
        verbose: Print progress lines. Turn off for runs inside a pool, where
            the interleaved output of several runs is unreadable anyway.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)

    set_global_seeds(cfg.seed)
    env = make_env(cfg, render_mode=None)
    discretizer = build_discretizer(cfg)
    agent = QLearningAgent(discretizer, int(env.action_space.n), config=cfg)

    if verbose:
        print(discretizer.describe())
        print(
            f"Reward {cfg.reward_preset} | Regel {cfg.algo} | alpha "
            f"{cfg.learning_rate} | gamma {cfg.gamma} | Seed {cfg.seed} | "
            f"Lauf {out_dir}"
        )

    train_logger = CsvLogger(out_dir / "train.csv", TRAIN_LOG_FIELDS)
    eval_logger = CsvLogger(out_dir / "eval.csv", EVAL_LOG_FIELDS)

    recent_returns: deque = deque(maxlen=50)
    recent_scores: deque = deque(maxlen=50)
    # kept for the sample-efficiency metric at the end, which the frame limit
    # can't censor
    episode_steps: List[int] = []
    episode_scores: List[float] = []
    best_score = -np.inf
    best_path = out_dir / "best.npz"
    best_truncation = 0.0
    recent_evals: deque = deque(maxlen=3)
    episode = 0
    started = time.time()

    obs, _ = env.reset(seed=cfg.seed)
    ep_return, ep_length, ep_td_sum = 0.0, 0, 0.0
    epsilon, alpha = agent.epsilon(0), agent.learning_rate(0)

    for step in range(1, cfg.total_steps + 1):
        epsilon = agent.epsilon(step)
        alpha = agent.learning_rate(step)
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # only `terminated` ends the bootstrap. A truncation means the frame
        # limit was hit: the episode stops, but the future wasn't worthless,
        # so it must not be learned as a death.
        ep_td_sum += abs(agent.update(obs, action, reward, next_obs, terminated, step))
        if truncated and not terminated:
            agent.finish_truncated_episode(next_obs, step)

        obs = next_obs
        ep_return += reward
        ep_length += 1

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
                    "alpha": round(alpha, 5),
                    "td_error": round(ep_td_sum / max(ep_length, 1), 5),
                    "coverage": round(agent.coverage, 4),
                }
            )
            if verbose and episode % cfg.log_interval == 0:
                elapsed = time.time() - started
                print(
                    f"Schritt {step:>8} | Episode {episode:>6} | "
                    f"Return(50) {np.mean(recent_returns):7.2f} | "
                    f"Score(50) {np.mean(recent_scores):6.2f} | "
                    f"eps {epsilon:.3f} | Abdeckung {agent.coverage:.2f} | "
                    f"{step / max(elapsed, 1e-9):.0f} Schritte/s"
                )
            obs, _ = env.reset()
            ep_return, ep_length, ep_td_sum = 0.0, 0, 0.0

        if step % cfg.eval_interval == 0:
            stats = quick_eval(agent, cfg, cfg.eval_episodes, seed=cfg.seed + 10_000)
            eval_logger.log({"step": step, **stats})
            # median, not mean: the score distribution has a long right tail,
            # so one lucky episode moves the mean more than an actually better
            # policy does
            recent_evals.append(stats["median_score"])
            # judge a checkpoint by the mean of the last few evals, not a
            # single one: the max over dozens of noisy draws is biased upwards
            # and would pick the luckiest eval instead of the best policy
            smoothed = float(np.mean(recent_evals))
            if verbose:
                censored = " [ZENSIERT]" if stats["truncation_rate"] >= 0.5 else ""
                print(
                    f"  Eval @ {step}: Median {stats['median_score']:.1f} "
                    f"(Mittel {stats['mean_score']:.1f}){censored} | "
                    f"geglaettet {smoothed:.1f} | max {stats['max_score']} | "
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
            agent.save(out_dir / "latest.npz", step=step)

    agent.save(out_dir / "latest.npz", step=cfg.total_steps)

    # a finished run must hold a best.npz, also when the budget isn't a
    # multiple of eval_interval and the last stretch was never measured
    if cfg.total_steps % cfg.eval_interval != 0:
        stats = quick_eval(agent, cfg, cfg.eval_episodes, seed=cfg.seed + 10_000)
        eval_logger.log({"step": cfg.total_steps, **stats})
        recent_evals.append(stats["median_score"])
        smoothed = float(np.mean(recent_evals))
        if verbose:
            print(
                f"  Abschluss-Eval @ {cfg.total_steps}: Median "
                f"{stats['median_score']:.1f} | geglaettet {smoothed:.1f}"
            )
        if smoothed > best_score:
            best_score = smoothed
            best_truncation = stats["truncation_rate"]
            agent.save(best_path, step=cfg.total_steps, eval_stats=stats)

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
        # smoothed median of the periodic evals, see the eval block
        "best_eval_score": float(best_score),
        "best_eval_truncation_rate": float(best_truncation),
        "train_score_last_50": float(np.mean(recent_scores)) if recent_scores else 0.0,
        # tabular-specific: how much of the table the run actually filled
        "discretizer": cfg.discretizer,
        "n_states": discretizer.n_states,
        "coverage": agent.coverage,
        "discretizer_description": discretizer.describe(),
        **steps_to_thresholds(episode_steps, episode_scores),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if verbose:
        print(
            f"Fertig in {minutes:.1f} min | bester Eval-Score {best_score:.2f} | "
            f"Abdeckung {agent.coverage:.2f}"
            + (
                "  (zensiert durch das Frame-Limit — echte Zahl via evaluate.py)"
                if summary["best_eval_truncation_rate"] >= 1.0
                else ""
            )
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a tabular Q-learning agent on FlappyBird-v0."
    )
    parser.add_argument("--run-name", default=QLearningConfig().run_name)
    parser.add_argument("--out-dir", default="runs", help="where runs are stored")
    add_config_arguments(parser)
    return parser


def config_from_args(args: argparse.Namespace) -> QLearningConfig:
    """Turns parsed arguments into a config.

    Every flag is named after its field, so ``from_dict`` does the mapping.
    """
    return QLearningConfig.from_dict(vars(args))


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    train(cfg, Path(args.out_dir) / cfg.run_name)


if __name__ == "__main__":
    main()
