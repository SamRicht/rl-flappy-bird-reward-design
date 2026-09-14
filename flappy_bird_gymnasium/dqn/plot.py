"""Plots learning curves from one or more training runs.

Examples:
    python -m flappy_bird_gymnasium.dqn.plot runs/dqn_v1
    python -m flappy_bird_gymnasium.dqn.plot runs/dqn_v1 runs/dqn_no_double
"""

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> dict:
    """Reads a CSV into a dict of float columns.

    Empty cells become NaN rather than being dropped, so that every column
    stays aligned with the `step` column (matplotlib skips NaNs when drawing).
    """
    columns = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                columns.setdefault(key, []).append(
                    np.nan if value in ("", None) else float(value)
                )
    return {k: np.asarray(v) for k, v in columns.items()}


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_runs(run_dirs: List[Path], window: int = 50, out: Optional[Path] = None):
    """Draws score, return, loss and eval score for every run."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax_score, ax_return), (ax_loss, ax_eval) = axes

    for run_dir in run_dirs:
        label = run_dir.name
        train = _read_csv(run_dir / "train.csv")
        offset = window - 1 if len(train["score"]) >= window else 0

        ax_score.plot(
            train["step"][offset:], _moving_average(train["score"], window), label=label
        )
        ax_return.plot(
            train["step"][offset:], _moving_average(train["return"], window), label=label
        )
        if "loss" in train:
            # one loss value per episode (the last learning step of it)
            ax_loss.plot(train["step"], train["loss"], label=label, alpha=0.7)

        eval_path = run_dir / "eval.csv"
        if eval_path.exists():
            evaluation = _read_csv(eval_path)
            ax_eval.plot(
                evaluation["step"], evaluation["mean_score"], marker="o", label=label
            )

    ax_score.set(
        title=f"Training score (moving avg, {window} episodes)",
        xlabel="environment steps",
        ylabel="score",
    )
    ax_return.set(
        title=f"Training return (moving avg, {window} episodes)",
        xlabel="environment steps",
        ylabel="return",
    )
    ax_loss.set(
        title="TD loss (last learning step of each episode)",
        xlabel="environment steps",
        ylabel="huber loss",
    )
    ax_loss.set_yscale("log")
    ax_eval.set(
        title="Greedy evaluation", xlabel="environment steps", ylabel="mean score"
    )
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()

    if out is not None:
        fig.savefig(out, dpi=150)
        print(f"saved {out}")
    else:
        plt.show()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot DQN learning curves.")
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--window", type=int, default=50, help="moving-average window")
    parser.add_argument("--out", type=Path, default=None, help="save instead of show")
    args = parser.parse_args(argv)
    plot_runs(args.runs, window=args.window, out=args.out)


if __name__ == "__main__":
    main()
