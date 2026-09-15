"""Plots learning curves from one or more training runs.

Examples:
    python -m flappy_bird_gymnasium.dqn.plot runs/dqn_v1
    python -m flappy_bird_gymnasium.dqn.plot runs/dqn_v1 runs/dqn_no_double
"""

import argparse
import csv
import json
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


def _label(run_dir: Path) -> str:
    """Names a run by its directory, plus its reward scheme when one is known."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return run_dir.name
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    preset = config.get("reward_preset")
    return f"{run_dir.name} ({preset})" if preset else run_dir.name


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_runs(run_dirs: List[Path], window: int = 50, out: Optional[Path] = None):
    """Draws the learning diagnostics of every run into one figure.

    Returns of different reward schemes are not comparable with each other --
    that is the point of changing the reward. The *score* panels are, which is
    why they come first.
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    (ax_score, ax_eval), (ax_return, ax_length), (ax_loss, ax_flap) = axes

    for run_dir in run_dirs:
        label = _label(run_dir)
        train = _read_csv(run_dir / "train.csv")
        offset = window - 1 if len(train["score"]) >= window else 0
        steps = train["step"][offset:]

        ax_score.plot(steps, _moving_average(train["score"], window), label=label)
        ax_return.plot(steps, _moving_average(train["return"], window), label=label)
        ax_length.plot(steps, _moving_average(train["length"], window), label=label)
        if "flap_rate" in train:
            ax_flap.plot(steps, _moving_average(train["flap_rate"], window), label=label)
        if "loss" in train:
            # one loss value per episode (the last learning step of it)
            ax_loss.plot(train["step"], train["loss"], label=label, alpha=0.7)

        eval_path = run_dir / "eval.csv"
        if eval_path.exists():
            evaluation = _read_csv(eval_path)
            ax_eval.plot(
                evaluation["step"], evaluation["mean_score"], marker="o", label=label
            )

    avg = f"moving avg, {window} episodes"
    ax_score.set(
        title=f"Training score ({avg})", xlabel="environment steps", ylabel="score"
    )
    ax_eval.set(
        title="Greedy evaluation", xlabel="environment steps", ylabel="mean score"
    )
    ax_return.set(
        title=f"Training return ({avg}) -- scheme-specific",
        xlabel="environment steps",
        ylabel="return",
    )
    ax_length.set(
        title=f"Episode length ({avg})", xlabel="environment steps", ylabel="frames"
    )
    ax_loss.set(
        title="TD loss (last learning step of each episode)",
        xlabel="environment steps",
        ylabel="huber loss",
    )
    ax_loss.set_yscale("log")
    ax_flap.set(
        title=f"Flap rate ({avg})", xlabel="environment steps", ylabel="flaps per frame"
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


def _curve_on_grid(
    run_dir: Path, grid: np.ndarray, window: int, column: str
) -> Optional[np.ndarray]:
    """Rolling mean of one run's `column`, resampled onto a shared step grid.

    Runs finish different numbers of episodes at different steps, so their
    curves cannot be averaged elementwise. Interpolating each onto the same
    grid first is what makes a mean across seeds meaningful.
    """
    train = _read_csv(run_dir / "train.csv")
    if column not in train or len(train[column]) < window:
        return None
    smoothed = _moving_average(train[column], window)
    steps = train["step"][window - 1 :]
    return np.interp(grid, steps, smoothed, left=np.nan, right=np.nan)


def plot_study(
    study_root: Path,
    window: int = 50,
    out: Optional[Path] = None,
    column: str = "score",
) -> None:
    """Draws one band per variant: mean across seeds, shaded by one std.

    This is the figure a comparison is argued from. The band is the point --
    two variants whose bands overlap for their whole length have not been
    shown to differ, however far apart their means happen to end up.
    """
    variants: dict = {}
    for config_path in sorted(study_root.glob("*/config.json")):
        run_dir = config_path.parent
        variants.setdefault(run_dir.name.rsplit("_seed", 1)[0], []).append(run_dir)
    if not variants:
        raise SystemExit(f"keine Laeufe unter {study_root}")

    # a shared grid ending at the shortest run, so no curve is extrapolated
    last_steps = []
    for run_dirs in variants.values():
        for run_dir in run_dirs:
            train = _read_csv(run_dir / "train.csv")
            if len(train.get("step", [])):
                last_steps.append(train["step"][-1])
    grid = np.linspace(0, min(last_steps), 300)

    fig, (ax_curve, ax_final) = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    finals, labels = [], []
    for index, (variant, run_dirs) in enumerate(sorted(variants.items())):
        curves = [_curve_on_grid(d, grid, window, column) for d in run_dirs]
        curves = [c for c in curves if c is not None]
        if not curves:
            continue
        stacked = np.vstack(curves)
        # grid points before the first episode of a run are NaN for that run;
        # a std needs at least two of them, so compute it only where it exists
        counts = np.sum(~np.isnan(stacked), axis=0)
        mean = np.full(stacked.shape[1], np.nan)
        std = np.zeros(stacked.shape[1])
        present, usable = counts > 0, counts > 1
        if present.any():
            mean[present] = np.nanmean(stacked[:, present], axis=0)
        if usable.any():
            std[usable] = np.nanstd(stacked[:, usable], axis=0, ddof=1)
        color = colors[index % len(colors)]

        ax_curve.plot(grid, mean, color=color, label=f"{variant} (n={len(curves)})")
        ax_curve.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18)

        finals.append((np.nanmean(mean[-20:]), np.nanmean(std[-20:])))
        labels.append(variant)

    ax_curve.set(
        title=f"Trainings-{column} je Variante (gleitender Mittelwert ueber {window} Episoden)",
        xlabel="Umgebungsschritte",
        ylabel=column,
    )
    ax_curve.grid(alpha=0.3)
    ax_curve.legend()

    positions = np.arange(len(labels))
    ax_final.bar(
        positions,
        [f[0] for f in finals],
        yerr=[f[1] for f in finals],
        capsize=5,
        color=[colors[i % len(colors)] for i in range(len(labels))],
    )
    ax_final.set_xticks(positions)
    ax_final.set_xticklabels(labels, rotation=20, ha="right")
    ax_final.set(title="Endniveau (letzte 20 Gitterpunkte)", ylabel=column)
    ax_final.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150)
        print(f"gespeichert: {out}")
    else:
        plt.show()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot DQN learning curves.")
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--window", type=int, default=50, help="moving-average window")
    parser.add_argument("--out", type=Path, default=None, help="save instead of show")
    parser.add_argument(
        "--study",
        action="store_true",
        help="treat the single argument as a study root and band its seeds",
    )
    parser.add_argument(
        "--column",
        default="score",
        help="study mode: which train.csv column to band (score, length, ...)",
    )
    args = parser.parse_args(argv)
    if args.study:
        plot_study(args.runs[0], window=args.window, out=args.out, column=args.column)
    else:
        plot_runs(args.runs, window=args.window, out=args.out)


if __name__ == "__main__":
    main()
