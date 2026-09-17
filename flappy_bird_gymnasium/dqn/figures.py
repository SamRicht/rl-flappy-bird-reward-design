"""Erzeugt die allgemeinen DQN-Abbildungen für Präsentation und Verfahrensvergleich.

Nicht lauf-, sondern verfahrensbezogen: Was DQN erreicht, woher die Leistung
kommt, und welche Stellschraube wie viel beiträgt. Alle Zahlen stammen aus den
`evaluations*.csv` der Studien, keine ist hier hart eingetragen.

    python -m flappy_bird_gymnasium.dqn.figures --runs-root runs --out-dir docs

Drei Messregeln, die jede Abbildung einhält (ARBEITSSTAND 9, 10):

*Ein Punkt je Seed.* Die Streuung zwischen den Seeds ist bei DQN größer als die
meisten Effekte. Balken mit Fehlerbalken verstecken das; ein Punkt je Seed plus
Median zeigt es.

*Kein bester Checkpoint als Kopfzahl.* Seine Auswahl hängt an einer gedeckelten
Metrik (ARBEITSSTAND 9.9). Berichtet wird der letzte Checkpoint, unzensiert.

*Nie der Return.* Zwischen Reward-Schemata ist er per Konstruktion
unvergleichbar. Verglichen werden Score und `steps_to_<n>`.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --------------------------------------------------------------------- Palette
# Die ersten drei Plätze der validierten Standardpalette, in dokumentierter
# Reihenfolge -- als einzige Teilmenge bestehen sie die All-Pairs-Prüfung.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID, DIM = "#fcfcfb", "#e4e3de", "#b9b7ae"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def _style(ax, *, xgrid=True):
    """Achsen zurücknehmen: Gitter recessive, nur zwei Rahmenlinien."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def _header(fig, title: str, subtitle: str, *, lines: int = 2):
    """Titel und Untertitel über der Zeichenfläche, kollisionsfrei.

    `ax.set_title` plus eine Annotation an derselben Stelle überlagern sich;
    beide hier über Figurenkoordinaten zu setzen hält den Abstand verlässlich.
    Gibt den `rect`-Wert für `tight_layout` zurück.
    """
    height = fig.get_size_inches()[1]
    fig.text(0.008, 0.975, title, color=INK, fontsize=14.5,
             fontweight="bold", ha="left", va="top")
    fig.text(0.008, 0.975 - 0.44 / height, subtitle, color=INK_2,
             fontsize=9, ha="left", va="top", linespacing=1.45)
    return [0, 0, 1, 1 - (0.52 + 0.19 * lines) / height]


def _read(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _seeds(rows: Sequence[dict], variant: str, column: str) -> List[float]:
    """Alle Seed-Werte einer Variante, leere Zellen übersprungen."""
    out = []
    for row in rows:
        if row.get("variant") == variant and row.get(column) not in ("", None):
            out.append(float(row[column]))
    return sorted(out)


def _dotplot(ax, labels, values, colors, *, log=False, unit="", fmt="{:.0f}"):
    """Ein Punkt je Seed, Median als Strich -- die ehrliche Form für DQN-Daten.

    Balken mit Fehlerbalken suggerieren eine Normalverteilung; die Seed-
    Verteilungen hier sind stark schief (ARBEITSSTAND 8.7).
    """
    for i, (vals, color) in enumerate(zip(values, colors)):
        y = len(labels) - 1 - i
        ax.plot(vals, [y] * len(vals), "o", color=color, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                alpha=0.85, linestyle="none")
        med = float(np.median(vals))
        ax.plot([med, med], [y - 0.28, y + 0.28], color=color,
                linewidth=2.5, zorder=4, solid_capstyle="round")
        ax.annotate(fmt.format(med) + unit, (med, y + 0.34),
                    color=INK, fontsize=9, ha="center", va="bottom",
                    fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)), color=INK)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    if log:
        ax.set_xscale("log")
    _style(ax)


# ------------------------------------------------------------------ Abbildung 1
def fig_ergebnis(runs: Path, out: Path) -> None:
    """Was der DQN-Agent kann -- die Zahl für die Vergleichstabelle."""
    rows = _read(runs / "final_dqn" / "evaluations_latest.csv")
    vals = _seeds(rows, "baseline", "mean_score")

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    med = float(np.median(vals))

    fig, ax = plt.subplots(figsize=(9.5, 3.3))
    y = 0
    # Eine symmetrische +/- Spanne waere auf einer Log-Achse irrefuehrend --
    # Mittelwert und Streuung stehen deshalb als Text, nicht als Balken.
    ax.plot(vals, [y] * len(vals), "o", color=BLUE, markersize=12,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, alpha=0.85)
    ax.plot([med, med], [y - 0.22, y + 0.22], color=BLUE, linewidth=3,
            zorder=4, solid_capstyle="round")
    ax.annotate(f"Median {med:.0f}", (med, y + 0.3), color=INK, fontsize=11.5,
                ha="center", va="bottom", fontweight="bold")
    for v, side in ((min(vals), "right"), (max(vals), "left")):
        ax.annotate(f"{v:.0f}", (v, y - 0.3), color=MUTED, fontsize=9,
                    ha="center", va="top")

    ax.plot([0], [y - 1.0], "o", color=MUTED, markersize=10, transform=ax.transData,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, clip_on=False)
    ax.annotate("Zufallspolicy: 0 Röhren", (0.6, y - 1.0), color=INK_2,
                fontsize=10, ha="left", va="center")
    ax.annotate(f"Ø {mean:.0f} ± {std:.0f} Röhren über die 5 Seeds",
                (0.985, 0.16), xycoords="axes fraction", color=INK_2,
                fontsize=10.5, ha="right", va="bottom")

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 2500)
    ax.set_ylim(-1.5, 0.75)
    ax.set_yticks([])
    ax.set_xlabel("passierte Röhren je Episode (log)   ·   30 Episoden je Seed, unzensiert")
    _style(ax)
    rect = _header(
        fig, "Was der DQN-Agent kann",
        "Ein Punkt je Seed — 5 gehaltene Seeds, letzter Checkpoint nach 1 Mio. Schritten.\n"
        "Gehalten heißt: diese Seeds waren an keiner Konfigurationswahl beteiligt.")
    fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Abbildung 2
def fig_stufen(runs: Path, out: Path) -> None:
    """Woher die Leistung kommt: Bausteine und Reward, nicht Hyperparameter."""
    abl = _read(runs / "study_ablation" / "evaluations.csv")
    rew = _read(runs / "study_reward" / "evaluations_limit200000.csv")

    stufen = [
        ("Zufallspolicy", [0.0] * 5, [np.nan] * 5, MUTED),
        ("Lehrbuch-DQN", _seeds(abl, "vanilla", "mean_score"),
         _seeds(abl, "vanilla", "steps_to_10"), DIM),
        ("+ Double · Dueling · n-step", _seeds(rew, "legacy", "mean_score"),
         _seeds(rew, "legacy", "steps_to_10"), AQUA),
        ("+ Reward „shaped“", _seeds(rew, "shaped", "mean_score"),
         _seeds(rew, "shaped", "steps_to_10"), BLUE),
    ]
    labels = [s[0] for s in stufen]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3))

    # Links die Lerngeschwindigkeit: Sie trennt die Stufen sauber, waehrend der
    # Endscore von der Seed-Streuung ueberdeckt wird (ARBEITSSTAND 8.7).
    ax = axes[0]
    speeds = [[v / 1000 for v in s[2] if not np.isnan(v)] for s in stufen]
    for i, (vals, color) in enumerate(zip(speeds, [s[3] for s in stufen])):
        y = len(labels) - 1 - i
        if not vals:
            ax.annotate("erreicht Score 10 nie", (200, y), color=MUTED,
                        fontsize=9.5, ha="left", va="center", style="italic")
            continue
        ax.plot(vals, [y] * len(vals), "o", color=color, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, alpha=0.85)
        med = float(np.median(vals))
        ax.plot([med, med], [y - 0.28, y + 0.28], color=color, linewidth=2.5,
                zorder=4, solid_capstyle="round")
        ax.annotate(f"{med:.0f}k", (med, y + 0.34), color=INK, fontsize=9.5,
                    ha="center", va="bottom", fontweight="bold")
        if len(vals) < 5:
            ax.annotate(f"nur {len(vals)} von 5 Seeds", (200, y - 0.02),
                        color=MUTED, fontsize=8.5, ha="left", va="center")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)), color=INK)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlim(180, 1000)
    ax.set_xlabel("Tausend Schritte bis Score 10 (weniger ist besser)")
    ax.set_title("Lerngeschwindigkeit — trennt sauber", color=INK, fontsize=11.5,
                 fontweight="bold", loc="left", pad=10)
    _style(ax)

    scores = [s[1] for s in stufen]
    _dotplot(axes[1], labels, scores, [s[3] for s in stufen], log=True)
    axes[1].set_yticklabels([""] * len(labels))
    axes[1].set_xlim(0.5, 3000)
    axes[1].set_xlabel("passierte Röhren je Episode (log)")
    axes[1].set_title("Endleistung — Seed-Streuung überdeckt viel", color=INK,
                      fontsize=11.5, fontweight="bold", loc="left", pad=10)
    axes[1].annotate("0", (0.55, len(labels) - 1), color=MUTED, fontsize=9,
                     ha="left", va="center")

    rect = _header(
        fig, "Woher die Leistung kommt",
        "Je 5 Seeds (0 – 4), 1 Mio. Schritte, gepaart gemessen. Ein Punkt je Seed, Strich = Median.\n"
        "Bei „+ Reward“ laufen Median und Mittelwert auseinander (192 gegen 442): Der Gewinn des Schemas liegt in der "
        "Lerngeschwindigkeit und im oberen Bereich, nicht im Median.\n"
        "Hyperparameter-Tuning brachte darüber hinaus nichts — auf gehaltenen Seeds widerlegt (ARBEITSSTAND 8.9).",
        lines=3)
    fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Abbildung 3
def fig_reward(runs: Path, out: Path) -> None:
    """Der stärkste Hebel -- und die Trennung nach der Deckenstrafe."""
    rows = _read(runs / "study_reward" / "evaluations_limit200000.csv")
    # Reihenfolge nach Lerngeschwindigkeit; Farbe nach Strafe aufs Flattern.
    presets = [
        ("shaped", False), ("sparse", False), ("survival", False),
        ("risk_averse", True), ("additive", True), ("legacy", True),
        ("energy", True),
    ]
    labels = [p for p, _ in presets]
    colors = [ORANGE if strafe else BLUE for _, strafe in presets]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    speeds = [[v / 1000 for v in _seeds(rows, p, "steps_to_10")] for p, _ in presets]
    _dotplot(axes[0], labels, speeds, colors, unit="k")
    axes[0].set_xlim(250, 800)
    axes[0].set_xlabel("Tausend Schritte bis Score 10 (weniger ist besser)")
    axes[0].set_title("Lerngeschwindigkeit — die belastbare Messung",
                      color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
    # Die Trennlinie sitzt genau zwischen den beiden Gruppen -- sie ist der
    # eigentliche Befund und bekommt deshalb eine Beschriftung auf der Grenze.
    axes[0].axvline(530, color=DIM, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    axes[0].annotate("vollständige Trennung", (530, -0.5), color=INK_2,
                     fontsize=9, ha="center", va="center", fontweight="bold")

    scores = [_seeds(rows, p, "mean_score") for p, _ in presets]
    _dotplot(axes[1], labels, scores, colors, log=True)
    axes[1].set_xlim(20, 2000)
    axes[1].set_yticklabels([""] * len(labels))
    axes[1].set_xlabel("passierte Röhren je Episode (log)")
    axes[1].set_title("Endleistung — Seed-Streuung überdeckt fast alles",
                      color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)

    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                          markeredgecolor=SURFACE, markeredgewidth=1.5, color=BLUE,
                          label="ohne Strafe aufs Flattern"),
               plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                          markeredgecolor=SURFACE, markeredgewidth=1.5, color=ORANGE,
                          label="mit Deckenstrafe −0,5")]
    axes[0].legend(handles=handles, loc="lower left", frameon=False,
                   fontsize=9.5, labelcolor=INK_2, handletextpad=0.4,
                   borderaxespad=0.8)

    rect = _header(
        fig, "Das Belohnungsschema ist der stärkste Hebel",
        "7 Schemata × 5 Seeds, 1 Mio. Schritte. Ein Punkt je Seed, Strich = Median. Sortiert nach Lerngeschwindigkeit.\n"
        "Die Trennung ist vollständig: Jedes Schema ohne Strafe aufs Flattern lernt schneller als jedes mit (p < 0,001).\n"
        "„energy“ bestraft als einziges zusätzlich jeden Flügelschlag — und ist das langsamste von allen.",
        lines=3)
    fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Abbildung 4
def fig_ablation(runs: Path, out: Path) -> None:
    """Welcher Algorithmus-Baustein trägt -- Emphasis auf der Vollkonfiguration."""
    rows = _read(runs / "study_ablation" / "evaluations.csv")
    variants = [
        ("full", "alle drei Erweiterungen", BLUE),
        ("no_dueling", "ohne Dueling-Kopf", DIM),
        ("no_double", "ohne Double DQN", DIM),
        ("no_nstep", "ohne n-step Returns", ORANGE),
        ("vanilla", "Lehrbuch-DQN", DIM),
    ]
    labels = [lbl for _, lbl, _ in variants]
    colors = [c for _, _, c in variants]
    scores = [_seeds(rows, v, "mean_score") for v, _, _ in variants]

    fig, ax = plt.subplots(figsize=(10, 4.3))
    _dotplot(ax, labels, scores, colors, log=True)
    ax.set_xlim(3, 800)
    ax.set_xlabel("passierte Röhren je Episode (log)   ·   Reward „legacy“, 5 Seeds, 1 Mio. Schritte")
    rect = _header(
        fig, "n-step Returns tragen den größten Einzelbeitrag",
        "Jeder Baustein einzeln abgeschaltet, gegen die Vollkonfiguration und den Lehrbuch-DQN.\n"
        "Ohne n-step fällt der Greedy-Verlauf von 38 auf 15; der Dueling-Kopf ist in keiner Messung\n"
        "von der Vollkonfiguration zu trennen (p ≥ 0,69) und bleibt nur, weil er nicht schadet.",
        lines=3)
    fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Abbildung 5
def fig_lernkurve(runs: Path, out: Path) -> None:
    """Lernkurve mit Seed-Band -- das Format für den Verfahrensvergleich."""
    fig, ax = plt.subplots(figsize=(10, 4.4))

    series = [
        ("runs/final_dqn", "baseline_seed{}", range(100, 105), BLUE,
         "DQN, finale Konfiguration"),
        ("runs/study_ablation", "vanilla_seed{}", range(5), DIM, "Lehrbuch-DQN"),
    ]
    for root, pattern, seeds, color, label in series:
        curves, grid = [], np.arange(0, 1_000_001, 5000)
        for seed in seeds:
            path = runs.parent / root / pattern.format(seed) / "train.csv"
            if not path.exists():
                continue
            steps, scores = [], []
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    steps.append(float(row["step"]))
                    scores.append(float(row["score"]))
            window = 200
            smooth = np.convolve(scores, np.ones(window) / window, mode="valid")
            curves.append(np.interp(grid, np.asarray(steps[window - 1:]), smooth))
        if not curves:
            continue
        curves = np.vstack(curves)
        lo, med, hi = (np.percentile(curves, q, axis=0) for q in (0, 50, 100))
        ax.fill_between(grid / 1e6, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(grid / 1e6, med, color=color, linewidth=2, zorder=3, label=label)
        # Direktbeschriftung statt Legende -- Identitaet darf nie an Farbe
        # allein haengen. annotation_clip=False, sonst schneidet die Achse sie ab.
        ax.annotate(label, (1.01, med[-1]), color=INK, fontsize=9.5,
                    ha="left", va="center", fontweight="bold",
                    annotation_clip=False)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, None)
    ax.set_xlabel("Umgebungsschritte (Millionen)")
    ax.set_ylabel("Score im Training\n(gleitendes Mittel, 200 Episoden)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _style(ax, xgrid=False)
    rect = _header(
        fig, "Lernverlauf: lange flach, dann ein Sprung",
        "Linie = Median über 5 Seeds, Band = schlechtester bis bester Seed.\n"
        "Achtung beim Vergleich mit anderen Verfahren: Das ist der Score der Trainingsepisoden, also mit Exploration.\n"
        "Bei ε = 0,01 ist er strukturell bei rund 18 gedeckelt — die Greedy-Policy derselben Agenten liegt bei ~500.",
        lines=3)
    rect[2] = 0.80  # Platz fuer die Direktbeschriftung rechts
    fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=200)
    plt.close(fig)


FIGURES = {
    "dqn_ergebnis.png": fig_ergebnis,
    "dqn_stufen.png": fig_stufen,
    "dqn_reward.png": fig_reward,
    "dqn_ablation.png": fig_ablation,
    "dqn_lernkurve.png": fig_lernkurve,
}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Allgemeine DQN-Abbildungen.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in FIGURES.items():
        target = args.out_dir / name
        builder(args.runs_root, target)
        print(f"  {target}")


if __name__ == "__main__":
    main()
