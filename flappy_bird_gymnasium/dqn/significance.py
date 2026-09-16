"""Tests whether a difference between two variants survives the seed noise.

A study gives every variant a handful of seeds, and the spread between those
seeds is large: in the ablation study the same configuration scored between 103
and 209 pipes depending on nothing but the seed. Reading the means alone
therefore invites conclusions the data does not carry -- in the reward study
`sparse` beat `legacy` by 242 to 170 and still came out at p = 0.42.

The test used here is an **exact permutation test** on the Mann-Whitney U
statistic: it enumerates every way the measured values could be split between
the two variants and asks how often chance alone produces a separation at least
as extreme as the observed one. With 5 seeds against 5 the smallest attainable
p-value is 0.008, reached exactly when every seed of one variant beats every
seed of the other. Nothing is assumed about the distribution, which matters
because scores are heavily skewed by the occasional lucky run.

Two cautions that belong next to every number this prints:

* **Many comparisons.** A study with a dozen variants against one baseline finds
  something at p < 0.05 now and then without any real effect. A finding counts
  when it is large and shows up in several measurements, not when one p-value
  happens to be small.
* **This is not the only evidence.** `steps_to_<n>` and the greedy curve are
  usually the more sensitive metrics, because the final score carries the full
  oscillation of the policy.

Usage::

    python -m flappy_bird_gymnasium.dqn.significance runs/study_params
    python -m flappy_bird_gymnasium.dqn.significance runs/study_reward \
        --baseline legacy --file evaluations_limit200000.csv
"""

import argparse
import csv
import itertools
from math import comb
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

#: Compared for every variant. The final score is the headline number; the two
#: sample-efficiency metrics cannot be censored by the frame limit and are
#: usually the sharper instrument.
METRICS = ("mean_score", "steps_to_5", "steps_to_10")

#: Beyond this many splits the exact enumeration is replaced by sampling. 5 vs 5
#: needs 252, 8 vs 8 needs 12.870 -- the cap only bites for large studies.
EXACT_LIMIT = 200_000


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> float:
    """The U statistic: how often a value of `a` exceeds one of `b`, ties half."""
    return float(sum((x > y) + 0.5 * (x == y) for x in a for y in b))


def permutation_p(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    """Two-sided p-value for "a and b come from the same distribution".

    Enumerates every split when that is affordable, otherwise samples. The
    sampled variant adds one to numerator and denominator, which keeps the
    p-value from ever being reported as exactly zero.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.concatenate([a, b])
    centre = len(a) * len(b) / 2
    observed = abs(mann_whitney_u(a, b) - centre)

    splits = comb(len(pooled), len(a))
    if splits <= EXACT_LIMIT:
        extreme = sum(
            abs(mann_whitney_u(pooled[list(idx)], np.delete(pooled, list(idx))) - centre)
            >= observed - 1e-9
            for idx in itertools.combinations(range(len(pooled)), len(a))
        )
        return extreme / splits

    rng = np.random.default_rng(seed)
    draws = 20_000
    extreme = sum(
        abs(mann_whitney_u(perm[: len(a)], perm[len(a) :]) - centre) >= observed - 1e-9
        for perm in (rng.permutation(pooled) for _ in range(draws))
    )
    return (extreme + 1) / (draws + 1)


def read_runs(path: Path) -> Dict[str, Dict[str, List[float]]]:
    """Reads an `evaluations*.csv` into ``{variant: {metric: [per seed]}}``."""
    variants: Dict[str, Dict[str, List[float]]] = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if row["variant"] == "random":
            continue  # the reference row has no seeds to compare
        metrics = variants.setdefault(row["variant"], {metric: [] for metric in METRICS})
        for metric in METRICS:
            value = row.get(metric)
            # a level a run never reached is empty, not zero -- carried through
            # as NaN so it cannot silently count as "reached instantly"
            metrics[metric].append(float(value) if value else np.nan)
    return variants


def compare(
    variants: Dict[str, Dict[str, List[float]]], baseline: str
) -> List[Dict[str, object]]:
    """Every variant against the baseline, one row each."""
    if baseline not in variants:
        raise SystemExit(
            f"Basis {baseline!r} nicht gefunden; vorhanden: {sorted(variants)}"
        )
    rows = []
    for name in sorted(variants):
        if name == baseline:
            continue
        row: Dict[str, object] = {"variant": name, "seeds": len(variants[name]["mean_score"])}
        for metric in METRICS:
            reference = np.asarray(variants[baseline][metric], dtype=float)
            values = np.asarray(variants[name][metric], dtype=float)
            usable = ~np.isnan(reference).any() and not np.isnan(values).any()
            row[metric] = float(np.nanmean(values))
            row[f"{metric}_ref"] = float(np.nanmean(reference))
            row[f"{metric}_p"] = permutation_p(values, reference) if usable else None
        rows.append(row)
    return rows


def print_table(rows: List[Dict[str, object]], baseline: str) -> None:
    header = (
        f"{'Variante':<28}{'Score':>9}{'p':>8}"
        f"{'bis 5':>9}{'p':>8}{'bis 10':>9}{'p':>8}"
    )
    print(f"\nJede Variante gegen {baseline!r} (exakter Permutationstest)")
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: -r["mean_score"]):
        cells = [f"{row['variant']:<28}{row['mean_score']:>9.1f}{_p(row['mean_score_p']):>8}"]
        for metric in ("steps_to_5", "steps_to_10"):
            steps = row[metric]
            shown = "-" if np.isnan(steps) else f"{steps / 1000:.0f}k"
            cells.append(f"{shown:>9}{_p(row[f'{metric}_p']):>8}")
        print("".join(cells))
    reference = rows[0]
    print(
        f"{baseline:<28}{reference['mean_score_ref']:>9.1f}{'':>8}"
        f"{reference['steps_to_5_ref'] / 1000:>8.0f}k{'':>8}"
        f"{reference['steps_to_10_ref'] / 1000:>8.0f}k"
    )
    print(
        "\np = Wahrscheinlichkeit, dass allein die Seed-Streuung einen so grossen\n"
        "    Unterschied erzeugt. Bei 5 gegen 5 Seeds ist 0.008 der kleinste\n"
        "    moegliche Wert -- er bedeutet: jeder Seed der einen Variante liegt\n"
        "    vor jedem Seed der anderen.\n"
        "Achtung: Viele Varianten gegen dieselbe Basis liefern auch ohne echten\n"
        "    Effekt gelegentlich kleine p-Werte. Ein Befund zaehlt, wenn er gross\n"
        "    ist und in mehreren Messungen auftaucht -- nicht wegen eines p-Werts."
    )


def _p(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Seed-Streuung gegen Effekt testen.")
    parser.add_argument("study_root", type=Path)
    parser.add_argument(
        "--baseline",
        default="baseline",
        help="Variante, gegen die verglichen wird (z. B. baseline, legacy, full)",
    )
    parser.add_argument(
        "--file",
        default="evaluations.csv",
        help="welche Messung; z. B. evaluations_latest.csv oder evaluations_limit200000.csv",
    )
    args = parser.parse_args(argv)

    path = args.study_root / args.file
    if not path.exists():
        raise SystemExit(f"{path} fehlt -- zuerst summarize.py laufen lassen")
    rows = compare(read_runs(path), args.baseline)
    print_table(rows, args.baseline)


if __name__ == "__main__":
    main()
