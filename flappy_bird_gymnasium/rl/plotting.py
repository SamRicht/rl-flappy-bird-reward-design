"""Figures for the reward-design study.

Reads the CSV logs a study wrote and produces the figures used in the report:

``learning_curves``
    Score over environment steps, one line per variant, averaged over seeds with
    a band showing the spread between them.
``diagnostics``
    The PPO health indicators -- entropy, approximate KL, clip fraction and
    explained variance -- which say *why* a configuration learns or stalls.
``behaviour``
    Final flap rate and distance from the gap centre, i.e. how the reward shaped
    the policy rather than how well it scored.

Usage::

    python -m flappy_bird_gymnasium.rl.plotting runs/reward_study --out figures/
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Categorical palette, assigned to variants in a fixed order (never cycled).
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def _style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str = "") -> None:
    """Applies the recessive grid / muted axis treatment used by every figure."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)


def read_runs(runs_root: Path) -> Dict[str, List[Path]]:
    """Groups a study's run directories by variant.

    Run directories are named ``<variant>_seed<n>``; the variant is everything
    before the final ``_seed`` separator.

    Args:
        runs_root: Directory a study wrote its runs into.

    Returns:
        Mapping from variant name to its per-seed run directories, sorted.
    """
    variants: Dict[str, List[Path]] = {}
    for path in sorted(runs_root.iterdir()):
        if not (path / "progress.csv").exists():
            continue
        variant = path.name.rsplit("_seed", 1)[0]
        variants.setdefault(variant, []).append(path)
    return variants


def read_column(run_dir: Path, column: str, source: str = "progress.csv") -> tuple:
    """Reads one numeric column plus the matching ``global_step`` column.

    Rows in which the column is empty -- a progress row from an update during
    which no episode finished -- are skipped.

    Returns:
        Tuple of ``(steps, values)`` as float arrays.
    """
    steps: List[float] = []
    values: List[float] = []
    with (run_dir / source).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get(column, "")
            if raw in ("", "nan"):
                continue
            steps.append(float(row["global_step"]))
            values.append(float(raw))
    return np.asarray(steps), np.asarray(values)


def _resample(steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolates a run onto a shared step grid so seeds can be averaged."""
    return np.interp(grid, steps, values)


def aggregate_variant(
    run_dirs: Sequence[Path],
    column: str,
    n_points: int = 200,
    smooth: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Averages a column over the seeds of one variant.

    Args:
        run_dirs: The per-seed directories of the variant.
        column: Column to aggregate.
        n_points: Resolution of the shared step grid.
        smooth: Width of a centred moving average; ``1`` disables smoothing.

    Returns:
        Tuple ``(grid, median, low, high)`` where low/high are the 25th and 75th
        percentile across seeds.  The interquartile range is used rather than the
        min/max envelope, whose width grows with the number of seeds and would
        make an eight-seed study look noisier than a three-seed one.  The centre
        line is the median for the same reason: single runaway seeds are common
        here and would drag a mean around.
    """
    series = [read_column(run_dir, column) for run_dir in run_dirs]
    series = [(s, v) for s, v in series if len(s) > 1]
    if not series:
        raise ValueError(f"no usable data for column {column!r}")

    last = min(s[-1] for s, _ in series)
    grid = np.linspace(0, last, n_points)
    stacked = np.stack([_resample(s, v, grid) for s, v in series])

    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        stacked = np.stack([np.convolve(row, kernel, mode="same") for row in stacked])
        # The convolution tapers the ends; trim them rather than show the dip.
        edge = smooth // 2
        grid, stacked = grid[edge : -edge or None], stacked[:, edge : -edge or None]

    return (
        grid,
        np.median(stacked, axis=0),
        np.percentile(stacked, 25, axis=0),
        np.percentile(stacked, 75, axis=0),
    )


def plot_learning_curves(
    runs_root: Path,
    out_path: Path,
    column: str = "ep_score_mean",
    ylabel: str = "Pipes passed per episode",
    title: str = "Learning progress by reward scheme (median, IQR across seeds)",
    order: Optional[Sequence[str]] = None,
) -> Path:
    """Plots the learning curve of every variant of a study."""
    variants = read_runs(runs_root)
    names = [n for n in (order or sorted(variants)) if n in variants]

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    for index, name in enumerate(names):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        grid, mean, low, high = aggregate_variant(variants[name], column)
        ax.fill_between(grid, low, high, color=color, alpha=0.12, linewidth=0)
        ax.plot(grid, mean, color=color, linewidth=2, label=name, zorder=3)

    _style_axes(ax, "Environment steps", ylabel, title)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M")
    )
    legend = ax.legend(
        frameon=False, fontsize=10, labelcolor=INK, ncol=2, loc="upper left"
    )
    legend.set_zorder(5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_diagnostics(
    runs_root: Path,
    out_path: Path,
    order: Optional[Sequence[str]] = None,
) -> Path:
    """Plots the four PPO health indicators as a small-multiple grid."""
    variants = read_runs(runs_root)
    names = [n for n in (order or sorted(variants)) if n in variants]

    panels = [
        ("entropy", "Policy entropy (nats)"),
        ("approx_kl", "Approximate KL per update"),
        ("clip_fraction", "Clipped fraction"),
        ("explained_variance", "Explained variance of V(s)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURFACE)

    for ax, (column, label) in zip(axes.flat, panels):
        for index, name in enumerate(names):
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            grid, mean, _, _ = aggregate_variant(variants[name], column, smooth=9)
            ax.plot(grid, mean, color=color, linewidth=1.8, label=name)
        _style_axes(ax, "Environment steps", label)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M")
        )

    axes.flat[0].legend(frameon=False, fontsize=9, labelcolor=INK, ncol=2)
    fig.suptitle("PPO diagnostics", color=INK, fontsize=13, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_behaviour(
    evaluations: Dict[str, Dict[str, float]],
    out_path: Path,
    order: Optional[Sequence[str]] = None,
) -> Path:
    """Plots the behavioural fingerprint of each final policy.

    Args:
        evaluations: Mapping from variant to the dictionary returned by
            :func:`flappy_bird_gymnasium.rl.evaluate.evaluate`.
        out_path: Where to write the figure.
        order: Explicit variant order.
    """
    names = [n for n in (order or sorted(evaluations)) if n in evaluations]
    metrics = [
        ("score_mean", "Pipes per episode", 1.0),
        ("flap_rate_mean", "Flap rate (share of steps)", 1.0),
        ("gap_offset_mean", "Distance from gap centre (screen heights)", 1.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor=SURFACE)
    positions = np.arange(len(names))

    for ax, (key, label, scale) in zip(axes, metrics):
        values = [evaluations[n].get(key, np.nan) * scale for n in names]
        colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(names))]
        ax.bar(positions, values, color=colors, width=0.68, zorder=3)
        _style_axes(ax, "", label)
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=30, ha="right", color=INK, fontsize=9)
        ax.grid(axis="x", visible=False)
        for x, value in zip(positions, values):
            if np.isfinite(value):
                ax.text(
                    x,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=INK,
                )

    fig.suptitle(
        "What the reward taught the policy",
        color=INK,
        fontsize=13,
        x=0.01,
        ha="left",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument(
        "--order", nargs="+", default=None, help="Explicit variant order."
    )
    parser.add_argument(
        "--evaluations",
        type=Path,
        default=None,
        help="JSON file mapping variant -> evaluation dict, for the "
        "behaviour figure.",
    )
    args = parser.parse_args(argv)

    written = [
        plot_learning_curves(
            args.runs_root, args.out / "learning_curves.png", order=args.order
        ),
        plot_diagnostics(
            args.runs_root, args.out / "diagnostics.png", order=args.order
        ),
    ]
    if args.evaluations:
        evaluations = json.loads(args.evaluations.read_text(encoding="utf-8"))
        written.append(
            plot_behaviour(evaluations, args.out / "behaviour.png", order=args.order)
        )
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
