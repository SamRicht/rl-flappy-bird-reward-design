"""Figures for a study: learning curves, behaviour, and single-run diagnostics.

Two rules, both from what the data actually looks like:

* Median with an interquartile band, not mean with standard deviation. The
  score distribution is strongly right-skewed, so mean +/- std describes a
  shape the data doesn't have and regularly produces bands below zero. The
  interquartile range also doesn't widen with the number of seeds, the way a
  min/max envelope does.
* One dot per seed, plus the median. A bar with an error bar hides how many
  runs there were and whether they agreed. With four or five seeds the
  individual points are the result, so a reader should see them.

Three figures:

``study`` (default)
    Learning curve per variant, plus a dot plot of where each seed ended.
``--behaviour``
    What the reward taught the finished policy: score, flap rate and
    position relative to the gap, one dot per seed, the random policy as a
    reference line. Reads the ``evaluations*.csv`` from ``summarize.py``.
``--runs``
    Diagnostics of single runs: any columns of any of their CSV logs, plus
    the greedy measurements from ``eval.csv`` where a method writes one.

Every method logs to its own file names. ``--algorithm`` takes them from that
method's ``RunLoader``; ``--source`` and ``--step-column`` set them by hand, e.g.
for the PPO diagnostics in ``progress.csv``.

Usage::

    python -m flappy_bird_gymnasium.rl.analysis.plot runs/study_reward \\
        --out docs/reward.png
    python -m flappy_bird_gymnasium.rl.analysis.plot runs/study_reward \\
        --behaviour --out docs/reward_behaviour.png
    python -m flappy_bird_gymnasium.rl.analysis.plot runs/study_reward --runs \\
        --panels score td_error coverage flap_rate --out docs/diag.png
    python -m flappy_bird_gymnasium.rl.analysis.plot runs/ppo_study --runs \\
        --source progress.csv --step-column global_step --window 1 \\
        --panels entropy approx_kl clip_fraction explained_variance
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from flappy_bird_gymnasium.rl.analysis.runs import (
    STEP_COLUMN,
    TRAIN_LOG,
    collect_runs,
    read_column,
    resolve_loader,
)

#: Categorical palette, assigned to variants in a fixed order so the same
#: variant keeps its colour across figures. The same palette the DQN and PPO
#: figures use, so figures from different methods fit next to each other.
COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"
REFERENCE = "#8a8880"

#: Training-log columns of the ``--runs`` figure when none are given. Every
#: method logs these; method-specific diagnostics are asked for by name.
DEFAULT_PANELS = ("score", "return", "length", "flap_rate")

#: Columns drawn on a log axis, since they span orders of magnitude.
LOG_SCALED = ("td_error", "loss")

#: What a log file holds, as ``(description, what one row counts)``.
#:
#: A curve read from a training log shows episodes played *with* exploration
#: and against the training frame limit; one read from ``eval.csv`` shows the
#: periodic greedy measurement. The two rank variants differently -- measured
#: on one tabular reward study, ``energy`` led the training score and came
#: fourth on the greedy one -- so every axis says which of them it shows.
#: Without that, two figures from the same study look like they contradict
#: each other.
SOURCE_KINDS: Dict[str, Tuple[str, str]] = {
    TRAIN_LOG: ("Training, mit Exploration", "Episoden"),
    "episodes.csv": ("Training, mit Exploration", "Episoden"),  # the PPO work
    "eval.csv": ("Greedy, ohne Exploration", "Messpunkte"),
}


def _source_kind(source: str) -> Tuple[str, str]:
    """How to describe ``source`` on an axis, and what one of its rows counts.

    An unregistered log (``progress.csv`` of the PPO work, say) is named by
    its file name alone rather than guessed at.
    """
    return SOURCE_KINDS.get(source, ("", "Zeilen"))


def _origin(source: str) -> str:
    """``source`` plus what it holds, for an axis label."""
    kind, _ = _source_kind(source)
    return f"{source} - {kind}" if kind else source


#: Panels of the ``--behaviour`` figure: column, axis label, and whether the
#: axis may turn logarithmic once the seeds spread over orders of magnitude.
BEHAVIOUR = (
    ("median_score", "Score (Median je Lauf)", True),
    ("mean_flap_rate", "Flap-Rate", False),
    ("mean_gap_offset", "Lage zur Lueckenmitte [px]\n+ = unterhalb", False),
    ("mean_abs_gap_offset", "|Abstand| zur Lueckenmitte [px]", False),
)


def _pyplot(out: Optional[Path]):
    """Imports pyplot, switching to a file-only backend when writing a file."""
    import matplotlib

    if out is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_MUTED,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "font.size": 9,
        }
    )
    return plt


def _millions(axis) -> None:
    """Environment steps as ``2M`` instead of ``2000000`` or ``1e6``."""
    from matplotlib.ticker import FuncFormatter

    axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))


def _finish(plt, figure, out: Optional[Path]) -> None:
    figure.tight_layout()
    if out is None:
        plt.show()
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, bbox_inches="tight")
        print(f"geschrieben: {out}")
    plt.close(figure)


def _ordered(names: Sequence[str], order: Optional[Sequence[str]]) -> List[str]:
    """``order`` first, as given, then the remaining names alphabetically."""
    if not order:
        return sorted(names)
    unknown = set(order) - set(names)
    if unknown:
        raise SystemExit(f"unknown variants {sorted(unknown)}; have {sorted(names)}")
    return list(order) + sorted(set(names) - set(order))


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean, trimmed so the ends aren't half-windows."""
    if values.size < window or window < 2:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def smoothed(
    run_dir: Path,
    column: str,
    window: int,
    source: str = TRAIN_LOG,
    step_column: str = STEP_COLUMN,
) -> Tuple[np.ndarray, np.ndarray]:
    """One run's column as a rolling mean, each value at the step it ends on."""
    steps, values = read_column(run_dir, column, source, step_column)
    curve = rolling(values, window)
    # named offset because black writes `steps[window - 1 :]` and flake8 then
    # complains about E203
    first = window - 1
    aligned = steps[first:] if curve.size != steps.size else steps
    return aligned, curve


def variant_band(
    run_dirs: Sequence[Path],
    column: str,
    window: int,
    points: int = 200,
    source: str = TRAIN_LOG,
    step_column: str = STEP_COLUMN,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Median and quartiles of ``column`` across the seeds of one variant.

    Each seed's curve is smoothed and then interpolated onto a shared step
    grid, because the runs end their episodes at different steps and can't
    be averaged row by row. The grid covers only the steps every seed has
    data for: ``np.interp`` holds a curve's last value flat beyond its end,
    which would quietly draw a shorter run as if it had plateaued.

    Returns:
        ``(grid, median, q25, q75)``, or None when no seed has enough data.
    """
    curves = []
    for run_dir in run_dirs:
        steps, values = smoothed(run_dir, column, window, source, step_column)
        if steps.size >= 2:
            curves.append((steps, values))
    if not curves:
        return None

    start = max(float(c[0][0]) for c in curves)
    end = min(float(c[0][-1]) for c in curves)
    if end <= start:
        return None
    grid = np.linspace(start, end, points)
    stacked = np.vstack([np.interp(grid, steps, values) for steps, values in curves])
    return (
        grid,
        np.median(stacked, axis=0),
        np.percentile(stacked, 25, axis=0),
        np.percentile(stacked, 75, axis=0),
    )


def _dots(axis, index: int, points: Sequence[float], color: str) -> None:
    """One dot per seed, spread sideways, with the median as a bar.

    The dots are drawn over the bar: a seed that sits exactly on the median
    must not disappear behind it.
    """
    if not points:
        return
    jitter = np.linspace(-0.12, 0.12, len(points)) if len(points) > 1 else [0.0]
    axis.scatter(
        [index + j for j in jitter],
        points,
        color=color,
        s=26,
        zorder=5,
        alpha=0.85,
        edgecolors=SURFACE,
        linewidths=0.8,
    )
    axis.plot(
        [index - 0.25, index + 0.25],
        [np.median(points)] * 2,
        color=color,
        linewidth=2.2,
        zorder=4,
        solid_capstyle="round",
    )


def _variant_axis(axis, names: Sequence[str]) -> None:
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=30, ha="right", color=INK, fontsize=8)
    axis.grid(axis="x", visible=False)


def plot_study(
    study_root: Path,
    column: str = "score",
    window: int = 200,
    out: Optional[Path] = None,
    order: Optional[Sequence[str]] = None,
    source: str = TRAIN_LOG,
    step_column: str = STEP_COLUMN,
) -> None:
    """Learning curves per variant plus a dot plot of the final level.

    Both halves read the same ``source``, and every axis names it: with the
    default training log they show episodes played *with* exploration, which
    is a different quantity from the greedy score ``--behaviour`` reports and
    can rank the variants differently. Passing ``--source eval.csv --column
    median_score`` draws the greedy measurement instead, which is the one that
    lines up with ``--behaviour``.
    """
    plt = _pyplot(out)
    grouped = collect_runs(study_root)
    if not grouped:
        raise SystemExit(f"no finished runs under {study_root}")
    names = _ordered(list(grouped), order)

    figure, (curve_ax, dot_ax) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": (2, 1)}
    )
    for index, variant in enumerate(names):
        color = COLORS[index % len(COLORS)]
        run_dirs = grouped[variant]
        band = variant_band(
            run_dirs, column, window, source=source, step_column=step_column
        )
        if band is not None:
            grid, median, low, high = band
            curve_ax.plot(
                grid, median, color=color, label=f"{variant} (n={len(run_dirs)})"
            )
            curve_ax.fill_between(grid, low, high, color=color, alpha=0.15, linewidth=0)
        # the same rolling mean the curve ends on, so both halves agree
        finals = []
        for run_dir in run_dirs:
            values = read_column(run_dir, column, source, step_column)[1]
            if values.size:
                finals.append(float(np.mean(values[-window:])))
        _dots(dot_ax, index, finals, color)

    kind, unit = _source_kind(source)
    origin = _origin(source)
    # window 1 means no smoothing at all; "gleitend ueber 1 Episoden" would be
    # both wrong and unreadable
    smoothed = f", gleitend ueber {window} {unit}" if window > 1 else ""
    final = f"Mittel der letzten {window} {unit}" if window > 1 else "letzter Messwert"

    curve_ax.set_xlabel("Umgebungsschritte")
    curve_ax.set_ylabel(f"{column}{smoothed}\n{origin}")
    curve_ax.set_title(
        f"Lernkurven {study_root.name}: {kind or source}\n"
        "Median und Quartile ueber die Seeds",
        loc="left",
        fontsize=10,
    )
    _millions(curve_ax)
    curve_ax.legend(frameon=False, fontsize=8)

    _variant_axis(dot_ax, names)
    dot_ax.set_ylabel(f"{column}, {final}\n{origin}")
    dot_ax.set_title("ein Punkt je Seed, Strich = Median", loc="left", fontsize=10)
    _finish(plt, figure, out)


def _read_evaluations(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _reference(axis, value: float, shown: Sequence[float]) -> None:
    """The random policy's value: a line if it's near the data, else a label.

    Random play sits far from any trained policy in flap rate and gap offset.
    Drawn as a line there, it would stretch the axis until the differences
    between the variants, the actual subject, vanish.
    """
    if value != value:
        return
    low, high = (min(shown), max(shown)) if shown else (value, value)
    margin = max(high - low, abs(high) * 0.1, 1e-9)
    if low - margin <= value <= high + margin:
        axis.axhline(value, color=REFERENCE, linewidth=1, linestyle=(0, (4, 3)))
        return
    # in the title row, where it can't cover a data point
    direction = "darueber" if value > high else "darunter"
    axis.set_title(
        f"Zufall: {value:.3g} ({direction})", loc="right", color=REFERENCE, fontsize=8
    )


def plot_behaviour(
    evaluations: Path,
    out: Optional[Path] = None,
    order: Optional[Sequence[str]] = None,
) -> None:
    """What each variant's finished policy does, one dot per seed.

    Two reward schemes can reach the same score while one hugs the centre of
    the gap and the other skims its edge; that difference is what a study of
    reward design is after, and the score alone doesn't show it.
    """
    plt = _pyplot(out)
    rows = _read_evaluations(evaluations)
    reference = next((r for r in rows if r["variant"] == "random"), None)
    names = _ordered(sorted({r["variant"] for r in rows} - {"random"}), order)
    if not names:
        raise SystemExit(f"no variants in {evaluations}")

    figure, axes = plt.subplots(1, len(BEHAVIOUR), figsize=(3.1 * len(BEHAVIOUR), 4))
    for axis, (column, label, symlog) in zip(axes, BEHAVIOUR):
        shown: List[float] = []
        for index, variant in enumerate(names):
            points = [
                float(r[column])
                for r in rows
                if r["variant"] == variant and r.get(column) not in ("", None)
            ]
            points = [p for p in points if p == p]
            _dots(axis, index, points, COLORS[index % len(COLORS)])
            shown += points
        if column == "mean_gap_offset":
            axis.axhline(0.0, color=INK_MUTED, linewidth=0.8)
        if symlog and shown and max(shown) > 100 * max(min(shown), 1.0):
            axis.set_yscale("symlog", linthresh=1)
        if reference is not None and reference.get(column) not in ("", None):
            _reference(axis, float(reference[column]), shown)
        axis.set_ylabel(label)
        _variant_axis(axis, names)

    figure.suptitle(
        f"Was der Reward der Policy beigebracht hat ({evaluations.name}); "
        "gestrichelt bzw. 'Zufall' = Zufallspolicy",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    _finish(plt, figure, out)


def plot_runs(
    run_dirs: Sequence[Path],
    panels: Sequence[str] = DEFAULT_PANELS,
    window: int = 200,
    out: Optional[Path] = None,
    source: str = TRAIN_LOG,
    step_column: str = STEP_COLUMN,
) -> None:
    """Diagnostics of single runs: columns of one log plus the greedy curve.

    The greedy panel reads ``eval.csv``. It is the curve without exploration
    noise, and usually the first place a stalled run shows. Left out for
    methods that don't write the file.
    """
    plt = _pyplot(out)
    greedy_runs = [d for d in run_dirs if (d / "eval.csv").exists()]
    count = len(panels) + (1 if greedy_runs else 0)
    rows = (count + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(11, 3 * rows), squeeze=False)
    flat = axes.ravel()

    for axis, column in zip(flat, panels):
        for index, run_dir in enumerate(run_dirs):
            steps, curve = smoothed(run_dir, column, window, source, step_column)
            if steps.size:
                axis.plot(
                    steps,
                    curve,
                    color=COLORS[index % len(COLORS)],
                    label=run_dir.name,
                    linewidth=1.2,
                )
        smoothing = f", gleitend ueber {window}" if window > 1 else ""
        axis.set_title(f"{column} ({_origin(source)}{smoothing})", loc="left")
        if column in LOG_SCALED:
            axis.set_yscale("log")
        _millions(axis)

    if greedy_runs:
        greedy = flat[len(panels)]
        for index, run_dir in enumerate(run_dirs):
            steps, values = read_column(run_dir, "median_score", "eval.csv")
            if not steps.size:
                steps, values = read_column(run_dir, "mean_score", "eval.csv")
            if steps.size:
                greedy.plot(
                    steps,
                    values,
                    color=COLORS[index % len(COLORS)],
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                )
        greedy.set_title("Greedy-Messung waehrend des Trainings (eval.csv)", loc="left")
        _millions(greedy)
    for axis in flat[count:]:
        axis.set_visible(False)

    flat[0].legend(frameon=False, fontsize=7)
    _finish(plt, figure, out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw the runs of a study.")
    parser.add_argument("target", type=Path, help="study directory, or a run directory")
    parser.add_argument("--column", default="score", help="log column (study)")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None, help="file instead of window")
    parser.add_argument("--order", nargs="+", default=None, help="variant order")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--runs",
        action="store_true",
        help="diagnostics of single runs instead of the study overview",
    )
    mode.add_argument(
        "--behaviour",
        action="store_true",
        help="behaviour of the finished policies, from summarize.py's output",
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        default=list(DEFAULT_PANELS),
        help="log columns for --runs",
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        help="take log file and step column from this method's RunLoader",
    )
    parser.add_argument(
        "--source", default=None, help=f"log file inside a run (default {TRAIN_LOG})"
    )
    parser.add_argument(
        "--step-column",
        default=None,
        help=f"step column of that log (default {STEP_COLUMN})",
    )
    parser.add_argument(
        "--file",
        default="evaluations.csv",
        help="which measurement --behaviour reads, inside the study directory",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    # explicit flags win over the loader, the loader over the defaults
    source, step_column = TRAIN_LOG, STEP_COLUMN
    if args.algorithm:
        loader = resolve_loader(args.algorithm)
        source, step_column = loader.train_log, loader.step_column
    source = args.source or source
    step_column = args.step_column or step_column

    if args.runs:
        run_dirs = (
            [args.target]
            if (args.target / source).exists()
            else [p for p in sorted(args.target.iterdir()) if (p / source).exists()]
        )
        if not run_dirs:
            raise SystemExit(f"no {source} under {args.target}")
        plot_runs(
            run_dirs,
            args.panels,
            window=args.window,
            out=args.out,
            source=source,
            step_column=step_column,
        )
    elif args.behaviour:
        plot_behaviour(args.target / args.file, out=args.out, order=args.order)
    else:
        plot_study(
            args.target,
            args.column,
            args.window,
            args.out,
            args.order,
            source=source,
            step_column=step_column,
        )


if __name__ == "__main__":
    main()
