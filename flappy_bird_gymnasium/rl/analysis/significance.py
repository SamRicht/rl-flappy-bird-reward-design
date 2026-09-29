"""Is a difference between two variants bigger than the noise between seeds?

Permutation test on the Mann-Whitney U statistic. It assumes nothing about the
shape of the distribution, which matters here: the score is strongly
right-skewed, so a t-test on the mean would test an assumption the data
doesn't meet.

With few seeds the test is enumerated exactly, the number of ways to split
``n`` against ``m`` values is small enough to walk through. That also means p
has a floor: five vs five seeds can never go below 2/252 = 0.0079, six vs six
not below 0.0022. A comparison that "fails" at four seeds may just be failing
because four seeds can't produce a smaller number.

A seed that never reached a ``steps_to_<n>`` level enters the test as
infinitely slow. The test only uses ranks, so that is exactly right: the seed
counts as worse than every seed that got there, without inventing a number of
steps. Dropping it instead would compare the variant on its lucky seeds only,
and skipping the test altogether would throw away the clearest result there
is: "all of these got there, none of those did".

Usage::

    python -m flappy_bird_gymnasium.rl.analysis.significance runs/study_reward \\
        --baseline legacy
    python -m flappy_bird_gymnasium.rl.analysis.significance runs/study_reward \\
        --baseline legacy --file evaluations_latest.csv --metric mean_score
"""

import argparse
import csv
import itertools
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Metrics compared by default. The median score is the headline because the
#: mean is dominated by the tail. ``steps_to_<n>`` is here because it stays
#: comparable once the score saturates, and because it has repeatedly
#: separated variants the score couldn't.
METRICS: Tuple[str, ...] = ("median_score", "steps_to_10", "steps_to_50")

#: Above this many arrangements the test is sampled instead of enumerated.
#: Five vs five needs 252, eight vs eight 12 870.
EXACT_LIMIT = 200_000
SAMPLES = 20_000

#: Guards the ``>=`` against floating-point noise in U. U itself is always a
#: multiple of 0.5, so this can never merge two genuinely different values.
_TOLERANCE = 1e-9


def is_censored_metric(metric: str) -> bool:
    """Whether an empty cell of ``metric`` means "never reached"."""
    return metric.startswith("steps_to_")


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> float:
    """Number of pairs in which ``a`` exceeds ``b``, ties counting a half."""
    return sum((x > y) + 0.5 * (x == y) for x in a for y in b)


def permutation_p(a: Sequence[float], b: Sequence[float], seed: int = 0) -> float:
    """Two-sided p of the observed U under relabelling.

    Exact while the enumeration is affordable. Otherwise sampled from the
    same null distribution, with one added to numerator and denominator so
    a sampled p is never reported as exactly zero.
    """
    a, b = list(a), list(b)
    if not a or not b:
        return float("nan")
    pool = a + b
    n = len(a)
    centre = len(a) * len(b) / 2
    observed = abs(mann_whitney_u(a, b) - centre) - _TOLERANCE

    total = math.comb(len(pool), n)
    if total <= EXACT_LIMIT:
        extreme = 0
        for indices in itertools.combinations(range(len(pool)), n):
            chosen = set(indices)
            left = [pool[i] for i in indices]
            right = [pool[i] for i in range(len(pool)) if i not in chosen]
            if abs(mann_whitney_u(left, right) - centre) >= observed:
                extreme += 1
        return extreme / total

    rng = random.Random(seed)
    extreme = 0
    for _ in range(SAMPLES):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        if abs(mann_whitney_u(shuffled[:n], shuffled[n:]) - centre) >= observed:
            extreme += 1
    return (extreme + 1) / (SAMPLES + 1)


def smallest_p(n: int, m: int) -> float:
    """The smallest two-sided p that ``n`` against ``m`` seeds can produce."""
    if not n or not m:
        return float("nan")
    return min(1.0, 2 / math.comb(n + m, n))


def read_runs(path: Path) -> Dict[str, Dict[str, List[float]]]:
    """Reads an ``evaluations*.csv`` into ``{variant: {metric: [per seed]}}``.

    An empty ``steps_to_<n>`` means the level was never reached and becomes
    ``inf``, see the module docstring. Reading it as zero would make the
    worst result look like the best. Empty cells of any other metric are
    dropped. The random reference is not a variant and is skipped.
    """
    out: Dict[str, Dict[str, List[float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant = row["variant"]
            if variant == "random":
                continue
            per_metric = out.setdefault(variant, {})
            for metric, value in row.items():
                if value in ("", None):
                    if is_censored_metric(metric):
                        per_metric.setdefault(metric, []).append(math.inf)
                    continue
                try:
                    per_metric.setdefault(metric, []).append(float(value))
                except ValueError:
                    continue
    return out


def compare(
    variants: Dict[str, Dict[str, List[float]]],
    baseline: str,
    metrics: Sequence[str] = METRICS,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Every variant against ``baseline``, one row each, the baseline first."""
    if baseline not in variants:
        raise SystemExit(
            f"unknown baseline {baseline!r}; available: {sorted(variants)}"
        )
    names = [baseline] + sorted(n for n in variants if n != baseline)
    rows: List[Dict[str, object]] = []
    for name in names:
        row: Dict[str, object] = {"variant": name}
        for metric in metrics:
            values = variants[name].get(metric, [])
            reference = variants[baseline].get(metric, [])
            row[metric] = _median(values)
            row[f"{metric}_n"] = len(values)
            if is_censored_metric(metric):
                row[f"{metric}_reached"] = sum(math.isfinite(v) for v in values)
            row[f"{metric}_p"] = (
                float("nan")
                if name == baseline or not values or not reference
                else permutation_p(reference, values, seed=seed)
            )
            row[f"{metric}_p_min"] = (
                float("nan")
                if name == baseline
                else smallest_p(len(reference), len(values))
            )
        rows.append(row)
    return rows


def _median(values: Sequence[float]) -> float:
    """Median that tolerates ``inf``; NaN for an empty list."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _shown(metric: str, value: float) -> str:
    if value != value:
        return "-"
    if math.isinf(value):
        return "nie"
    if is_censored_metric(metric):
        return f"{value / 1e6:.2f}M"
    return f"{value:.2f}"


def print_table(
    rows: Sequence[Dict[str, object]],
    baseline: str,
    metrics: Sequence[str] = METRICS,
) -> None:
    # plain ASCII throughout, see summarize.print_table
    header = f"{'Variante':22s}"
    for metric in metrics:
        header += f" {metric:>14s} {'p':>7s} {'n':>4s}"
    print(header)
    for row in rows:
        line = f"{str(row['variant']):22s}"
        for metric in metrics:
            value = float(row.get(metric, float("nan")))  # type: ignore[arg-type]
            p = float(row.get(f"{metric}_p", float("nan")))  # type: ignore[arg-type]
            n = row.get(f"{metric}_n", 0)
            reached = row.get(f"{metric}_reached")
            count = f"{reached}/{n}" if reached is not None and reached != n else n
            p_shown = "" if p != p else f"{p:.4f}"
            star = "*" if (p == p and p < 0.05) else " "
            line += f" {_shown(metric, value):>14s} {p_shown:>6s}{star} {count!s:>4}"
        print(line)

    comparisons = len(rows) - 1
    floors = sorted(
        {
            round(float(row[f"{metric}_p_min"]), 4)  # type: ignore[arg-type]
            for row in rows[1:]
            for metric in metrics
            if row.get(f"{metric}_p_min") == row.get(f"{metric}_p_min")
        }
    )
    print(
        f"\nReferenz: {baseline}. '*' = p < 0.05, zweiseitig. "
        "Median ueber die Seeds; 'nie' = mindestens\ndie Haelfte der Seeds "
        "erreichte die Schwelle nicht, n = erreicht/Seeds."
    )
    if floors:
        shown = " bzw. ".join(f"{f:.4f}" for f in floors)
        print(
            f"Kleinstes bei diesen Seed-Zahlen erreichbares p: {shown}. Ein "
            "Vergleich, der es\nerreicht, ist vollstaendig getrennt -- mehr "
            "Trennung gibt es nicht."
        )
    if comparisons > 1:
        print(
            f"Bei {comparisons} Vergleichen gegen dieselbe Referenz waere nach "
            f"Bonferroni p < {0.05 / comparisons:.4f} noetig."
        )
    print(
        "Ein Test ersetzt nicht das Hinsehen: Ein Befund zaehlt, wenn er gross "
        "ist und in mehreren\nMessungen auftaucht. Lernkurven und Streuung je "
        "Seed gehoeren daneben."
    )


def output_path(evaluations: Path, baseline: str) -> Path:
    """``evaluations_latest.csv`` -> ``significance_latest_vs_<baseline>.csv``."""
    stem = evaluations.stem
    prefix = "evaluations"
    suffix = stem.replace(prefix, "", 1) if stem.startswith(prefix) else f"_{stem}"
    return evaluations.with_name(f"significance{suffix}_vs_{baseline}.csv")


def write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Permutation test between the variants of a study."
    )
    parser.add_argument("study_root", type=Path)
    parser.add_argument(
        "--baseline",
        required=True,
        help="variant to compare against, e.g. legacy, baseline, basis",
    )
    parser.add_argument(
        "--file",
        default="evaluations.csv",
        help="which measurement inside study_root, e.g. evaluations_latest.csv",
    )
    parser.add_argument("--metric", nargs="+", default=list(METRICS))
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    path = args.study_root / args.file
    if not path.exists():
        raise SystemExit(f"{path} not found -- run summarize.py first")

    rows = compare(read_runs(path), args.baseline, args.metric, seed=args.seed)
    print_table(rows, args.baseline, args.metric)
    out = output_path(path, args.baseline)
    write_rows(out, rows)
    print(f"\ngeschrieben: {out}")


if __name__ == "__main__":
    main()
