"""Presentation figures for the factorial study, built on the shared chain.

Style, palette, the one-dot-per-seed plot, the learning-curve bands and the
reading of run directories all come from :mod:`flappy_bird_gymnasium.rl.analysis`,
so these figures look and count like every other figure of the project. What
is new here is only what a factorial grid needs and the shared chain does not
draw: main effects, a factor-by-factor heatmap, curves over ``q_init / V*``.

The figures carry no headline claims. Every panel is labelled with what it
shows -- axes, units, how many runs a point stands on -- so that it reads
without a caption, and the interpretation stays with whoever presents it.

Usage::

    python -m flappy_bird_gymnasium.qlearning.figures server_models/runs \\
        --out docs/evaluation/praesentation
    python -m flappy_bird_gymnasium.qlearning.figures server_models/runs --check

``--check`` draws nothing. It recomputes every number from the per-run rows,
compares the cell medians against the ``aggregate*.csv`` the shared
``summarize`` wrote independently, and prints the numbers a presentation
quotes into ``zahlen.txt``.
"""

import argparse
import csv
import re
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from flappy_bird_gymnasium.rl.analysis.plot import (
    COLORS,
    GRID,
    INK,
    INK_MUTED,
    REFERENCE,
    SURFACE,
    _dots,
    _millions,
    _pyplot,
    _reference,
    variant_band,
)
from flappy_bird_gymnasium.rl.analysis.runs import collect_runs

#: Mean reward per frame of each preset in ``rl/rewards.py``, with a pipe
#: every 37.8 frames (measured: 11 650 frames at a score of 308).
PIPE_EVERY = 37.8
FRAME_REWARD: Dict[str, float] = {
    "legacy": 0.1 + 0.9 / PIPE_EVERY,  # priority chain: the pipe replaces alive
    "additive": 0.1 + 1.0 / PIPE_EVERY,
    "risk_averse": 0.1 + 1.0 / PIPE_EVERY,
    "energy": 0.1 + 1.0 / PIPE_EVERY - 0.02 * 0.06,  # flap cost at ~6 % flaps
    "survival": 0.1,
    "sparse": 1.0 / PIPE_EVERY,
    "shaped": 1.0 / PIPE_EVERY,  # potential-based shaping telescopes
}

#: Dense schemes first, then the sparse ones; the same order in every figure.
REWARD_ORDER = [
    "legacy",
    "additive",
    "risk_averse",
    "energy",
    "survival",
    "sparse",
    "shaped",
]
SCHEME_COLOR = dict(zip(REWARD_ORDER, COLORS))

#: q_init at roughly 0.8 - 1.0 x V*: the grid value used wherever schemes are
#: compared "at their optimum".
CALIBRATED_Q = {
    "legacy": "10.0",
    "additive": "10.0",
    "risk_averse": "10.0",
    "energy": "10.0",
    "survival": "10.0",
    "sparse": "2.5",
    "shaped": "2.5",
}

#: Larger type for projection; everything else is the shared chain's style.
PROJECTION = {
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "axes.labelcolor": INK,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 150,
}
WIDE = (13.33, 7.5)


def achievable_value(preset: str, gamma: float = 0.99) -> float:
    """V*: the discounted value of surviving forever under ``preset``."""
    return FRAME_REWARD[preset] / (1.0 - gamma)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def newest_evaluations(study: Path) -> Path:
    """The measurement ``summarize`` wrote last; its name depends on the limit."""
    files = sorted(study.glob("evaluations*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no evaluations*.csv in {study} -- run summarize first")
    return files[-1]


def read_rows(path: Path) -> Tuple[List[Dict[str, str]], Optional[Dict[str, str]]]:
    """Per-run rows, and the random reference row if there is one."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    random_row = next((r for r in rows if r["variant"] == "random"), None)
    return [r for r in rows if r["variant"] != "random"], random_row


def values(rows: Sequence[Dict[str, str]], column: str) -> List[float]:
    out = []
    for row in rows:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            continue
        if value == value:
            out.append(value)
    return out


def where(rows, **conditions) -> List[Dict[str, str]]:
    return [r for r in rows if all(r.get(k) == v for k, v in conditions.items())]


def sweep_rows(root: Path, kind: str) -> List[Dict[str, str]]:
    """Rows of all ``gamma_coupled`` or ``nstep`` sweeps, with gamma attached."""
    out = []
    for study in sorted(root.glob(f"{kind}/*/study_sweep")):
        rows, _ = read_rows(newest_evaluations(study))
        match = re.search(r"_g([\d.]+)$", study.parent.name)
        for row in rows:
            row["gamma"] = match.group(1) if match else "0.99"
            out.append(row)
    return out


def _cell(rows, **conditions) -> float:
    """Median score of the runs matching ``conditions``."""
    return st.median(values(where(rows, **conditions), "median_score"))


def present(rows, candidates: Sequence[str] = ()) -> List[str]:
    """The reward presets that occur in ``rows``, in the fixed figure order.

    Read off the data rather than assumed: a sweep run for fewer schemes
    should give a smaller figure, not a crash.
    """
    found = {r["reward_preset"] for r in rows}
    order = list(candidates) or REWARD_ORDER
    return [p for p in order if p in found]


def ranks(data: Sequence[float]) -> np.ndarray:
    """Ranks with ties averaged, for a Spearman correlation."""
    data = np.asarray(data, dtype=float)
    order = data.argsort(kind="mergesort")
    raw = np.empty(len(data))
    raw[order] = np.arange(len(data))
    _, inverse, counts = np.unique(data, return_inverse=True, return_counts=True)
    return (np.bincount(inverse, weights=raw) / counts)[inverse]


def num(text: str) -> str:
    """'12.5' -> '12,5' for German axis labels."""
    return f"{float(text):g}".replace(".", ",")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _new(plt, ncols=1, sharey=False, size=WIDE):
    plt.rcParams.update(PROJECTION)
    return plt.subplots(1, ncols, figsize=size, sharey=sharey, squeeze=False)


def _save(plt, fig, out: Path) -> None:
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def _dots_big(axis, index: int, points: Sequence[float], color: str) -> None:
    """The shared chain's dot plot, scaled up for projection.

    ``_dots`` is sized for a report page; on a projected slide its dots and
    median bar disappear. Calling it and enlarging what it drew keeps the
    look identical without a second implementation.
    """
    # what _dots drew is whatever appeared after this point; named variables
    # because black writes the slice as "[n :]", which flake8 reports as E203
    first_collection, first_line = len(axis.collections), len(axis.lines)
    _dots(axis, index, points, color)
    for collection in list(axis.collections)[first_collection:]:
        collection.set_sizes([95])
    for line in list(axis.lines)[first_line:]:
        line.set_linewidth(3.6)


def _readable_reference(axis) -> None:
    """Enlarges the random-policy note ``_reference`` puts in the title row."""
    text = axis.get_title(loc="right")
    if text:
        text = text.replace("darueber", "darüber").replace(".", ",")
        axis.set_title(text, loc="right", fontsize=13, color=REFERENCE)


def _plain_log_ticks(axis) -> None:
    """Log axis labelled 20, 50, 100, 200 instead of 2x10^1.

    Ticks on the 1-2-5 sequence: matplotlib's default puts majors only on the
    decades, which leaves a single label when the data span less than one.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    low, high = axis.get_ylim()
    ticks = [
        m * 10**e for e in range(-1, 5) for m in (1, 2, 5) if low <= m * 10**e <= high
    ]
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.yaxis.set_minor_formatter(NullFormatter())


def fig_main_effects(plt, rows, out: Path) -> None:
    from matplotlib.lines import Line2D

    factors = [
        ("q_init", "q_init"),
        ("learning_rate_mode", "Schrittweiten-Modus"),
        ("discretizer", "Rasterisierung"),
        ("reward_preset", "Reward-Schema"),
    ]
    levels_of = {}
    for column, _ in factors:
        levels = sorted({r[column] for r in rows})
        if column == "q_init":
            levels.sort(key=float)
        if column == "reward_preset":
            levels = [p for p in REWARD_ORDER if p in levels]
        levels_of[column] = levels

    plt.rcParams.update(PROJECTION)
    fig, axes = plt.subplots(
        1,
        len(factors),
        figsize=(17, 7),
        sharey=True,
        gridspec_kw={
            "width_ratios": [max(len(levels_of[c]), 3.4) + 1 for c, _ in factors]
        },
    )
    for axis, (column, label) in zip(axes, factors):
        levels = levels_of[column]
        for i, level in enumerate(levels):
            scores = values(where(rows, **{column: level}), "median_score")
            q1, q3 = np.percentile(scores, [25, 75])
            axis.plot([i, i], [q1, q3], color=INK_MUTED, linewidth=2.2, zorder=2)
            axis.scatter(
                [i],
                [st.median(scores)],
                color=COLORS[0],
                s=90,
                zorder=3,
                edgecolors=SURFACE,
                linewidths=1.2,
            )
        # n is the same for every level of a factor: the grid is balanced
        n = len(where(rows, **{column: levels[0]}))
        axis.set_title(f"{label}\nn = {n} je Stufe", loc="left")
        axis.set_xticks(range(len(levels)))
        if column == "q_init":
            axis.set_xticklabels([num(lv) for lv in levels])
        elif column in ("reward_preset", "discretizer"):
            axis.set_xticklabels(levels, rotation=35, ha="right")
        else:
            axis.set_xticklabels(levels)
        axis.grid(axis="x", visible=False)
        axis.set_xlim(-0.6, len(levels) - 0.4)
    axes[0].set_ylabel("Score je Lauf\n(Median über 50 Episoden)")
    axes[0].legend(
        [
            Line2D([], [], marker="o", color=COLORS[0], linestyle="", markersize=9),
            Line2D([], [], color=INK_MUTED, linewidth=2.2),
        ],
        ["Median der Läufe", "25.–75. Perzentil"],
        loc="upper left",
        frameon=False,
    )
    _save(plt, fig, out)


def fig_heatmap(plt, rows, out: Path) -> None:
    import matplotlib
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    qs = sorted({r["q_init"] for r in rows}, key=float)
    grid = np.full((len(REWARD_ORDER), len(qs)), np.nan)
    for i, preset in enumerate(REWARD_ORDER):
        for j, q in enumerate(qs):
            scores = values(where(rows, reward_preset=preset, q_init=q), "median_score")
            if scores:
                grid[i, j] = st.median(scores)
    per_cell = len(where(rows, reward_preset=REWARD_ORDER[0], q_init=qs[0]))

    fig, axes = _new(plt, size=(12.5, 7.5))
    axis = axes[0][0]
    axis.grid(False)
    shown = np.clip(grid, 0.5, None)  # zeros would vanish on the log scale
    image = axis.imshow(
        shown,
        cmap="viridis",
        aspect="auto",
        norm=matplotlib.colors.LogNorm(vmin=0.5, vmax=np.nanmax(shown)),
    )
    for i in range(len(REWARD_ORDER)):
        for j in range(len(qs)):
            light = shown[i, j] < np.nanmax(shown) ** 0.6
            axis.text(
                j,
                i,
                f"{grid[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=14,
                color="white" if light else INK,
            )
        best = int(np.nanargmax(grid[i]))
        axis.add_patch(
            plt.Rectangle(
                (best - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=3
            )
        )
        position = np.interp(
            achievable_value(REWARD_ORDER[i]), [float(q) for q in qs], range(len(qs))
        )
        # only the lower part of the cell, so the number stays readable
        axis.plot(
            [position, position], [i + 0.18, i + 0.46], color=COLORS[7], linewidth=4
        )
    axis.set_xticks(range(len(qs)), [num(q) for q in qs])
    axis.set_yticks(range(len(REWARD_ORDER)), REWARD_ORDER)
    axis.set_xlabel("q_init")
    axis.set_ylabel("Reward-Schema")
    bar = fig.colorbar(image, ax=axis, pad=0.02)
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100) if t <= np.nanmax(shown)]
    bar.set_ticks(ticks)
    bar.set_ticklabels([str(t) for t in ticks])
    bar.minorticks_off()
    bar.set_label(
        f"Score je Lauf, Median über {per_cell} Läufe je Zelle\n"
        "(3 Raster × 2 Schrittweiten-Modi × 4 Seeds)"
    )
    axis.legend(
        [
            Line2D([], [], color=COLORS[7], linewidth=3.2),
            Patch(facecolor="none", edgecolor=INK_MUTED, linewidth=2),
        ],
        ["V* des Schemas (erreichbarer Wert)", "höchster Wert der Zeile"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        frameon=False,
    )
    _save(plt, fig, out)


def fig_qinit_relative(plt, rows, out: Path) -> None:
    from matplotlib.ticker import NullFormatter

    fig, axes = _new(plt, 2, sharey=True, size=(15, 7))
    for axis, mode in zip(axes[0], ("linear", "count")):
        for preset in REWARD_ORDER:
            v = achievable_value(preset)
            xs, ys = [], []
            for q in sorted({r["q_init"] for r in rows}, key=float):
                if float(q) == 0:
                    continue
                scores = values(
                    where(
                        rows, reward_preset=preset, q_init=q, learning_rate_mode=mode
                    ),
                    "median_score",
                )
                xs.append(float(q) / v)
                ys.append(st.median(scores))
            axis.plot(
                xs,
                ys,
                marker="o",
                color=SCHEME_COLOR[preset],
                linewidth=2.4,
                markersize=7,
                label=preset,
            )
        axis.axvline(
            1.0,
            color=REFERENCE,
            linewidth=2,
            linestyle=(0, (5, 3)),
            label="q_init = V*",
        )
        axis.set_xscale("log")
        axis.set_xticks([0.2, 0.3, 0.5, 0.8, 1, 2, 4, 8])
        axis.set_xticklabels(["0,2", "0,3", "0,5", "0,8", "1", "2", "4", "8"])
        # the log axis would print its minor ticks as 4x10^-1 on top of these
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("q_init / V*")
        axis.set_title(f"Schrittweiten-Modus {mode}", loc="left")
    axes[0][0].set_ylabel("Score je Lauf, Median über 12 Läufe\n(3 Raster × 4 Seeds)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    _save(plt, fig, out)


def fig_gamma(plt, sweep, out: Path) -> None:
    gammas = sorted({r["gamma"] for r in sweep}, key=float)
    colors = dict(zip(gammas, (COLORS[1], COLORS[0], COLORS[6])))
    presets = present(sweep)
    fig, axes = _new(plt, len(presets), size=(7.5 * len(presets), 7))
    for axis, preset in zip(axes[0], presets):
        for gamma in gammas:
            v = achievable_value(preset, float(gamma))
            subset = where(sweep, reward_preset=preset, gamma=gamma)
            points: Dict[float, List[float]] = {}
            for row in subset:
                ratio = round(float(row["q_init"]) / v, 1)
                points.setdefault(ratio, []).append(float(row["median_score"]))
            xs = sorted(points)
            axis.plot(
                xs,
                [st.median(points[x]) for x in xs],
                marker="o",
                color=colors[gamma],
                linewidth=2.6,
                markersize=8,
                label=f"γ = {num(gamma)}  (V* = {num(f'{v:.2f}')})",
            )
            for x in xs:
                axis.scatter(
                    [x] * len(points[x]),
                    points[x],
                    color=colors[gamma],
                    s=18,
                    alpha=0.45,
                    zorder=2,
                )
        axis.set_xticks([0.2, 0.5, 0.8, 1.0, 1.5])
        axis.set_xticklabels(["0,2", "0,5", "0,8", "1,0", "1,5"])
        axis.set_xlabel("q_init / V*  (V* je γ neu berechnet)")
        # single seeds reach 1 300 while the medians sit near 100
        axis.set_yscale("log")
        axis.autoscale_view()
        _plain_log_ticks(axis)
        axis.set_title(f"{preset}, Raster uniform, Modus linear", loc="left")
        axis.legend(frameon=False, loc="upper left")
    axes[0][0].set_ylabel(
        "Score je Lauf, log\n(Linie: Median über 4 Seeds, Punkte: Seeds)"
    )
    _save(plt, fig, out)


def fig_nstep(plt, sweep, out: Path) -> None:
    order = present(sweep, ("legacy", "survival", "sparse", "shaped")) or present(sweep)
    fig, axes = _new(plt, len(order), sharey=True, size=(15, 6.5))
    for axis, preset in zip(axes[0], order):
        subset = where(sweep, reward_preset=preset)
        steps = sorted({r["n_step"] for r in subset}, key=int)
        for i, n in enumerate(steps):
            _dots_big(
                axis,
                i,
                values(where(subset, n_step=n), "median_score"),
                SCHEME_COLOR[preset],
            )
        axis.set_xticks(range(len(steps)), [f"n={n}" for n in steps])
        axis.grid(axis="x", visible=False)
        q = subset[0]["q_init"]
        axis.set_title(f"{preset}  (q_init = {num(q)})", loc="left")
    axes[0][0].set_ylabel("Score je Lauf\n(Punkt: ein Seed, Strich: Median)")
    fig.supxlabel(
        "n_step (Länge des zurückgetragenen Returns), Raster uniform, " "Modus linear",
        fontsize=14,
        color=INK_MUTED,
    )
    _save(plt, fig, out)


def fig_behaviour(plt, rows, random_row, out: Path) -> None:
    panels = [
        ("median_score", "Score je Lauf (log)"),
        ("mean_flap_rate", "Flap-Rate\n(Anteil Frames mit Flügelschlag)"),
        ("mean_gap_offset", "Lage zur Lückenmitte [px]\n+ = unterhalb"),
    ]
    fig, axes = _new(plt, len(panels), size=(15, 6.8))
    for axis, (column, label) in zip(axes[0], panels):
        shown = []
        for i, preset in enumerate(REWARD_ORDER):
            points = values(
                where(
                    rows,
                    reward_preset=preset,
                    q_init=CALIBRATED_Q[preset],
                    discretizer="uniform",
                    learning_rate_mode="linear",
                ),
                column,
            )
            _dots_big(axis, i, points, SCHEME_COLOR[preset])
            shown += points
        if column == "median_score":
            axis.set_yscale("log")
            axis.autoscale_view()
            _plain_log_ticks(axis)
            if random_row is not None:
                axis.set_title(
                    f"Zufall: {float(random_row[column]):g}",
                    loc="right",
                    fontsize=13,
                    color=REFERENCE,
                )
        if column == "mean_gap_offset":
            axis.axhline(0, color=INK, linewidth=1)
        if random_row is not None and column != "median_score":
            _reference(axis, float(random_row[column]), shown)
            _readable_reference(axis)
        axis.set_xticks(range(len(REWARD_ORDER)))
        axis.set_xticklabels(REWARD_ORDER, rotation=35, ha="right")
        axis.grid(axis="x", visible=False)
        axis.set_ylabel(label)
    fig.supxlabel(
        "je Schema bei q_init ≈ 0,8–1,0 × V* (10 bzw. 2,5), Raster uniform, "
        "Modus linear; Punkt: ein Seed, Strich: Median",
        fontsize=13,
        color=INK_MUTED,
    )
    _save(plt, fig, out)


def fig_learning(plt, root: Path, out: Path) -> None:
    from matplotlib.lines import Line2D

    grouped = collect_runs(root / "study_matrix")
    qs = ["0.0", "2.5", "5.0", "10.0", "12.5", "20.0"]
    colors = dict(
        zip(qs, (REFERENCE, COLORS[6], COLORS[2], COLORS[0], COLORS[3], COLORS[7]))
    )
    fig, axes = _new(plt, 2, sharey=True, size=(15, 7))
    for axis, mode in zip(axes[0], ("linear", "count")):
        for q in qs:
            label = (
                f"reward_preset=legacy__q_init={q}__discretizer=uniform"
                f"__learning_rate_mode={mode}"
            )
            band = variant_band(
                grouped.get(label, []), "median_score", 1, source="eval.csv"
            )
            if band is None:
                continue
            grid, median, low, high = band
            axis.plot(
                grid, median, color=colors[q], linewidth=3.2 if q == "10.0" else 2.2
            )
            axis.fill_between(grid, low, high, color=colors[q], alpha=0.13, linewidth=0)
        _millions(axis)
        axis.set_xlabel("Trainingsschritte")
        axis.set_title(f"legacy, Raster uniform, Modus {mode}", loc="left")
    axes[0][0].set_ylabel(
        "Greedy-Zwischenmessung, Score\n"
        "(15 Episoden, Median über 4 Seeds, Band: Quartile)"
    )
    fig.legend(
        [Line2D([], [], color=colors[q], linewidth=3) for q in qs],
        [f"q_init = {num(q)}" + ("  (V* = 12,4)" if q == "12.5" else "") for q in qs],
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    _save(plt, fig, out)


def fig_efficiency(plt, rows, out: Path) -> None:
    from matplotlib.ticker import FuncFormatter

    thresholds = [
        "steps_to_1",
        "steps_to_5",
        "steps_to_10",
        "steps_to_25",
        "steps_to_50",
    ]
    fig, axes = _new(plt, 2, size=(15, 6.8))
    left, right = axes[0]
    reached = [len(values(rows, t)) for t in thresholds]
    left.bar(
        range(len(thresholds)),
        [r / len(rows) for r in reached],
        color=COLORS[0],
        width=0.6,
    )
    for i, count in enumerate(reached):
        left.text(
            i,
            count / len(rows) + 0.02,
            f"{count}/{len(rows)}",
            ha="center",
            fontsize=13,
        )
    left.set_xticks(
        range(len(thresholds)), [t.replace("steps_to_", "Score ") for t in thresholds]
    )
    left.set_ylim(0, 1.12)
    left.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.0%}".replace("%", " %"))
    )
    left.set_ylabel("Anteil der Läufe, die die Schwelle erreichen")
    left.set_xlabel(
        "gleitender Trainings-Score über 20 Episoden\n"
        "(mit ε = 0,01, Trainingslimit 3 000 Frames)"
    )
    left.grid(axis="x", visible=False)

    pairs = [
        (float(r["steps_to_10"]), float(r["median_score"]))
        for r in rows
        if r.get("steps_to_10") not in ("", None)
    ]
    xs, ys = map(np.asarray, zip(*pairs))
    rho = np.corrcoef(ranks(xs), ranks(ys))[0, 1]
    right.scatter(xs, np.clip(ys, 0.5, None), s=16, color=COLORS[0], alpha=0.45)
    right.set_yscale("log")
    right.autoscale_view()
    _plain_log_ticks(right)
    _millions(right)
    right.set_xlabel("Schritte bis gleitender Trainings-Score ≥ 10")
    right.set_ylabel("Score je Lauf (Endmessung, log)")
    # in the title row, where it cannot cover a point
    right.set_title(
        f"ein Punkt je Lauf, n = {len(xs)}, Spearman ρ = {rho:.2f}".replace(
            ".", ","
        ).replace("-", "−"),
        loc="right",
        fontsize=13,
        color=INK_MUTED,
    )
    _save(plt, fig, out)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def check(root: Path, report: Path) -> None:
    """Recomputes the quoted numbers and cross-checks them against summarize."""
    lines: List[str] = []
    say = lines.append
    problems = 0

    def compare(study: Path, label: str) -> None:
        nonlocal problems
        rows, _ = read_rows(newest_evaluations(study))
        aggregates = sorted(
            study.glob("aggregate*.csv"), key=lambda p: p.stat().st_mtime
        )
        if not aggregates:
            say(f"  {label}: keine aggregate*.csv zum Gegenpruefen")
            return
        with aggregates[-1].open(encoding="utf-8") as handle:
            agg = {r["variant"]: r for r in csv.DictReader(handle)}
        bad = 0
        for variant in {r["variant"] for r in rows}:
            mine = st.median(values(where(rows, variant=variant), "median_score"))
            theirs = float(agg[variant]["median_score"])
            if abs(mine - theirs) > 1e-6:
                bad += 1
        sizes = {len(where(rows, variant=v)) for v in {r["variant"] for r in rows}}
        trunc = sum(1 for t in values(rows, "truncation_rate") if t > 0)
        problems += bad
        say(
            f"  {label:34s} {len(rows):5d} Laeufe, Seeds je Zelle {sorted(sizes)}, "
            f"zensiert {trunc}, Abweichung zu aggregate: {bad}"
        )

    say("== Gegenpruefung gegen die aggregate*.csv aus summarize ==")
    compare(root / "study_matrix", "study_matrix")
    for study in sorted(root.glob("gamma_coupled/*/study_sweep")):
        compare(study, f"gamma {study.parent.name}")
    for study in sorted(root.glob("nstep/*/study_sweep")):
        compare(study, f"nstep {study.parent.name}")

    rows, random_row = read_rows(newest_evaluations(root / "study_matrix"))
    say("\n== Haupteffekte (Median der Laeufe) ==")
    for column in ("q_init", "learning_rate_mode", "discretizer", "reward_preset"):
        levels = sorted(
            {r[column] for r in rows},
            key=lambda x: float(x) if column == "q_init" else x,
        )
        cells = ", ".join(
            f"{lv} {st.median(values(where(rows, **{column: lv}), 'median_score')):.1f}"
            for lv in levels
        )
        say(f"  {column:20s} {cells}")

    say("\n== Optimum je Schema und Ausschnitt (Median je Zelle) ==")
    for name, cond in (
        ("gepoolt", {}),
        ("uniform+linear", {"discretizer": "uniform", "learning_rate_mode": "linear"}),
        ("uniform+count", {"discretizer": "uniform", "learning_rate_mode": "count"}),
    ):
        above = 0
        parts = []
        for preset in REWARD_ORDER:
            v = achievable_value(preset)
            meds = {
                q: st.median(
                    values(
                        where(rows, reward_preset=preset, q_init=q, **cond),
                        "median_score",
                    )
                )
                for q in sorted({r["q_init"] for r in rows}, key=float)
            }
            best = max(meds, key=meds.get)
            above += float(best) > v * (1 + 1e-9)
            parts.append(
                f"{preset} q={best} ({float(best) / v:.2f}xV*, {meds[best]:.1f})"
            )
        say(f"  {name}: oberhalb V* {above}/7")
        for part in parts:
            say(f"    {part}")

    say("\n== Verhalten am kalibrierten Optimum (uniform, linear) ==")
    for preset in REWARD_ORDER:
        sub = where(
            rows,
            reward_preset=preset,
            q_init=CALIBRATED_Q[preset],
            discretizer="uniform",
            learning_rate_mode="linear",
        )
        say(
            f"  {preset:12s} Score {st.median(values(sub, 'median_score')):7.1f}  "
            f"Flap {st.median(values(sub, 'mean_flap_rate')):.3f}  "
            f"Lage {st.median(values(sub, 'mean_gap_offset')):+6.1f}  "
            f"|Lage| {st.median(values(sub, 'mean_abs_gap_offset')):5.1f}"
        )
    if random_row:
        say(
            f"  {'Zufall':12s} Score {float(random_row['median_score']):7.1f}  "
            f"Flap {float(random_row['mean_flap_rate']):.3f}  "
            f"Lage {float(random_row['mean_gap_offset']):+6.1f}"
        )

    say("\n== Sample-Effizienz ==")
    for t in ("steps_to_1", "steps_to_5", "steps_to_10", "steps_to_25", "steps_to_50"):
        say(f"  {t:12s} erreicht {len(values(rows, t))}/{len(rows)}")
    pairs = [
        (float(r["steps_to_10"]), float(r["median_score"]))
        for r in rows
        if r.get("steps_to_10") not in ("", None)
    ]
    xs, ys = zip(*pairs)
    rho = np.corrcoef(ranks(xs), ranks(ys))[0, 1]
    say(f"  Spearman steps_to_10 vs Score: {rho:+.3f} (n={len(pairs)})")

    say("\n== gamma (gepoolt ueber die q_init-Verhaeltnisse) ==")
    sweep = sweep_rows(root, "gamma_coupled")
    for preset in present(sweep):
        cells = ", ".join(
            f"g={g} {_cell(sweep, reward_preset=preset, gamma=g):.1f}"
            for g in sorted({r["gamma"] for r in sweep}, key=float)
        )
        say(f"  {preset}: {cells}")

    say("\n== n_step (Median ueber 4 Seeds) ==")
    nsweep = sweep_rows(root, "nstep")
    for preset in present(nsweep):
        cells = ", ".join(
            f"n={n} {_cell(nsweep, reward_preset=preset, n_step=n):.1f}"
            for n in sorted({r["n_step"] for r in nsweep}, key=int)
        )
        say(f"  {preset}: {cells}")

    say("\n== bester Einzellauf ==")
    best = max(rows, key=lambda r: float(r["median_score"]))
    say(
        f"  {best['variant']} seed {best['seed']}: Median {best['median_score']}, "
        f"Mittel {best['mean_score']}, max {best['max_score']}"
    )

    say(f"\nAbweichungen insgesamt: {problems}")
    text = "\n".join(lines)
    print(text)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text + "\n", encoding="utf-8")
    print(f"\ngeschrieben: {report}")
    if problems:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Presentation figures of the study.")
    parser.add_argument(
        "root", type=Path, help="directory holding study_matrix, gamma_coupled, nstep"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/evaluation/praesentation")
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and cross-check the numbers, draw nothing",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.check:
        check(args.root, args.out / "zahlen.txt")
        return

    plt = _pyplot(args.out)
    rows, random_row = read_rows(newest_evaluations(args.root / "study_matrix"))
    fig_main_effects(plt, rows, args.out / "01_haupteffekte.png")
    fig_heatmap(plt, rows, args.out / "02_reward_x_qinit.png")
    fig_qinit_relative(plt, rows, args.out / "03_qinit_relativ_zu_vstar.png")
    fig_learning(plt, args.root, args.out / "04_lernverlauf_qinit.png")
    fig_behaviour(plt, rows, random_row, args.out / "05_verhalten.png")
    fig_gamma(plt, sweep_rows(args.root, "gamma_coupled"), args.out / "06_gamma.png")
    fig_nstep(plt, sweep_rows(args.root, "nstep"), args.out / "07_nstep.png")
    fig_efficiency(plt, rows, args.out / "08_sample_effizienz.png")


if __name__ == "__main__":
    main()
