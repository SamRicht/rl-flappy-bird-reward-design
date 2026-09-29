"""Measures every run of a study and aggregates the variants.

The training curve says how fast a config learned; this says what the finished
policy actually does. Three rules keep the comparison honest:

* **Paired.** Every run is measured on the same episode seeds, so all variants
  see identical pipe layouts and none wins by drawing easier ones.
* **Uncensored.** The frame limit defaults to each run's
  ``eval_max_episode_steps``, far above the training limit. Measured against
  the training limit, every good policy lands on the same capped score.
* **Anchored.** A random policy is measured through the same code path, so
  "better than chance" is a number and not an assumption.

The headline statistic is the median, across episodes and across seeds. The
score distribution has a long right tail: four seeds of one tabular config gave
mean scores of 51, 60, 49 and 455, while their medians stayed within a few
points of each other.

Usage::

    python -m flappy_bird_gymnasium.rl.analysis.summarize runs/study_reward \\
        --algorithm qlearning
    python -m flappy_bird_gymnasium.rl.analysis.summarize runs/study_reward \\
        --algorithm qlearning --checkpoint latest.npz --episodes 100
    # re-measure only the censored variants at a higher limit
    python -m flappy_bird_gymnasium.rl.analysis.summarize runs/study_reward \\
        --algorithm qlearning --variants shaped sparse --max-episode-steps 200000
"""

import argparse
import csv
import multiprocessing
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from flappy_bird_gymnasium.rl.analysis.runs import (
    LOADERS,
    LoadedRun,
    RunLoader,
    collect_runs,
    open_env,
    resolve_loader,
    sample_efficiency,
    unfinished_runs,
    variant_of,
)
from flappy_bird_gymnasium.rl.rollout import EVAL_SEED, rollout, summarize_rollout
from flappy_bird_gymnasium.rl.runlog import THRESHOLDS

#: Metrics condensed across the seeds of a variant, as median plus range.
AGGREGATED: Tuple[str, ...] = (
    "median_score",
    "mean_score",
    "max_score",
    "mean_length",
    "mean_flap_rate",
    "mean_gap_offset",
    "mean_abs_gap_offset",
    "truncation_rate",
    "mean_return",
)

#: Row keys that identify a single run and so never describe a variant.
_PER_RUN = ("variant", "seed", "run_dir")


def _rounded(stats: Dict[str, float]) -> Dict[str, object]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}


def evaluate_run(job: Dict[str, object]) -> Dict[str, object]:
    """Worker: measures one finished run and tags the result with its variant.

    The per-episode scores travel along under ``_scores`` for the pooled
    statistics of :func:`aggregate`. The leading underscore keeps them out of
    the CSV files.
    """
    loader: RunLoader = job["loader"]  # type: ignore[assignment]
    run_dir = Path(str(job["run_dir"]))
    loaded = loader.load(run_dir / str(job["checkpoint"]))
    config = loaded.config
    limit = int(job["max_episode_steps"] or config.eval_max_episode_steps)

    env = open_env(loaded, max_episode_steps=limit)
    measured = rollout(env, loaded.policy, int(job["episodes"]), int(job["seed"]))
    env.close()

    return {
        "algorithm": loader.algorithm,
        "variant": job["variant"],
        "seed": config.seed,
        "run_dir": str(run_dir),
        "checkpoint": job["checkpoint"],
        "reward_preset": config.reward_preset,
        "frame_limit": limit,
        **loaded.columns,
        **_rounded(summarize_rollout(measured)),
        **sample_efficiency(
            run_dir, source=loader.train_log, step_column=loader.step_column
        ),
        "_scores": list(measured["score"]),
    }


def random_baseline(
    loaded: LoadedRun, episodes: int, seed: int, max_episode_steps: int
) -> Dict[str, object]:
    """The same measurement for a policy that flaps at random.

    Goes through the same ``rollout`` as everything else. The DQN work once
    carried a hand-entered flap rate of 0.5 for this row; measured it is
    clearly below that, because a flap above the top of the screen has no
    effect.

    Played in the environment of ``loaded``, so under the same reward
    scheme, pipe gap and frame limit as the runs it is compared with.
    """
    rng = np.random.default_rng(0)
    env = open_env(loaded, max_episode_steps=max_episode_steps)
    measured = rollout(env, lambda obs: int(rng.integers(2)), episodes, seed)
    env.close()
    return {
        "algorithm": "random",
        "variant": "random",
        "seed": 0,
        "run_dir": "",
        "checkpoint": "",
        "reward_preset": loaded.config.reward_preset,
        "frame_limit": max_episode_steps,
        **_rounded(summarize_rollout(measured)),
        **{f"steps_to_{t}": None for t in THRESHOLDS},
        "_scores": list(measured["score"]),
    }


def aggregate(
    rows: Sequence[Dict[str, object]], metrics: Sequence[str] = AGGREGATED
) -> List[Dict[str, object]]:
    """Condenses the per-run rows into one row per variant.

    Reports the median across seeds with the full range next to it. The range
    isn't decoration: with this distribution it's usually wider than the
    difference between two variants, and a table that hides it invites
    reading noise as an effect.

    ``steps_to_<n>`` additionally carries how many seeds reached the level at
    all. A median over only the seeds that got there would flatter a variant
    that mostly didn't.

    Descriptive columns (algorithm, reward scheme, frame limit, table size,
    ...) are carried over when all seeds agree on them, and left out when
    they don't. A variant whose seeds were measured under different frame
    limits then has no ``frame_limit`` at all, instead of showing one of them.
    """
    efficiency = [f"steps_to_{t}" for t in THRESHOLDS]
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    out: List[Dict[str, object]] = []
    for variant, group in by_variant.items():
        entry: Dict[str, object] = {"variant": variant, "seeds": len(group)}
        for key, value in group[0].items():
            if key in _PER_RUN or key in metrics or key in efficiency:
                continue
            if key.startswith("_"):
                continue
            if all(r.get(key) == value for r in group):
                entry[key] = value

        for metric in metrics:
            values = [float(r[metric]) for r in group if _present(r.get(metric))]
            if not values:
                continue
            entry[metric] = round(statistics.median(values), 3)
            entry[f"{metric}_min"] = round(min(values), 3)
            entry[f"{metric}_max"] = round(max(values), 3)

        # every episode of every seed in one pool: the distribution-level
        # view, next to the median of per-seed medians above
        pooled = [s for r in group for s in r.get("_scores", [])]  # type: ignore
        if pooled:
            entry["pooled_median_score"] = float(np.median(pooled))
            entry["pooled_p90_score"] = float(np.percentile(pooled, 90))

        for key in efficiency:
            reached = [int(r[key]) for r in group if _present(r.get(key))]
            entry[key] = int(statistics.median(reached)) if reached else None
            entry[f"{key}_reached"] = f"{len(reached)}/{len(group)}"
        out.append(entry)
    # the reference goes last, where a reader looks for it
    return sorted(out, key=lambda e: (e["variant"] == "random", str(e["variant"])))


def _present(value: object) -> bool:
    """Whether a cell holds a number; ``None``, ``""`` and NaN don't."""
    if value is None or value == "":
        return False
    try:
        return not np.isnan(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def output_paths(
    study_root: Path,
    checkpoint: str,
    default_checkpoint: str,
    limit: Optional[int],
    variants: Optional[Sequence[str]] = None,
) -> Tuple[Path, Path]:
    """Filenames that encode what was measured and how.

    A second measurement with a different checkpoint, frame limit or subset
    of variants is a different number. Overwriting the first one with it is
    how two incomparable figures end up in the same report.
    """
    suffix = "" if checkpoint == default_checkpoint else f"_{Path(checkpoint).stem}"
    if limit:
        suffix += f"_limit{limit}"
    if variants:
        suffix += "_" + "+".join(sorted(variants))
    return (
        study_root / f"evaluations{suffix}.csv",
        study_root / f"aggregate{suffix}.csv",
    )


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Writes rows whose keys may differ, keeping every column.

    Keys starting with an underscore are in-memory only and left out.
    """
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames and not key.startswith("_"):
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def default_workers() -> int:
    """All cores but one: a study shouldn't make its machine unusable."""
    return max(1, (multiprocessing.cpu_count() or 2) - 1)


def evaluate_study(
    study_root: Path,
    loader: RunLoader,
    episodes: int = 50,
    seed: int = EVAL_SEED,
    checkpoint: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    workers: int = 0,
    with_baseline: bool = True,
    variants: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Measures every finished run of a study and returns (per run, per variant).

    Args:
        study_root: Directory holding the ``<variant>_seed<n>`` runs.
        loader: How to read this method's checkpoints.
        episodes: Episodes per run.
        seed: Base episode seed, shared by all runs.
        checkpoint: File name inside each run; the loader's default if None.
        max_episode_steps: Frame limit; each run's own
            ``eval_max_episode_steps`` if None.
        workers: Processes; 0 means all cores but one.
        with_baseline: Also measure the random policy.
        variants: Only these variants, e.g. to re-measure the censored ones
            at a higher limit. The result is then a different measurement and
            :func:`output_paths` files it under a different name.
    """
    checkpoint = checkpoint or loader.default_checkpoint
    grouped = collect_runs(study_root)
    if not grouped:
        raise SystemExit(f"no finished runs under {study_root}")
    if variants:
        unknown = set(variants) - set(grouped)
        if unknown:
            raise SystemExit(
                f"unknown variants {sorted(unknown)}; available: {sorted(grouped)}"
            )
        grouped = {k: v for k, v in grouped.items() if k in variants}

    for path in unfinished_runs(study_root):
        if not variants or variant_of(path) in variants:
            print(f"  unvollstaendig, uebersprungen: {path.name}")

    jobs: List[Dict[str, object]] = []
    for variant, run_dirs in grouped.items():
        for run_dir in run_dirs:
            if not (run_dir / checkpoint).exists():
                print(f"  kein {checkpoint}, uebersprungen: {run_dir.name}")
                continue
            jobs.append(
                {
                    "loader": loader,
                    "run_dir": str(run_dir),
                    "variant": variant,
                    "episodes": episodes,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "max_episode_steps": max_episode_steps,
                }
            )
    if not jobs:
        raise SystemExit(f"no run under {study_root} holds {checkpoint}")

    workers = workers if workers > 0 else default_workers()
    print(f"{len(jobs)} Laeufe x {episodes} Episoden auf {workers} Prozessen ...")
    rows: List[Dict[str, object]] = []

    def report(row: Dict[str, object]) -> None:
        print(
            f"  [{len(rows)}/{len(jobs)}] {Path(str(row['run_dir'])).name}: "
            f"Median {row['median_score']:.1f}, Mittel {row['mean_score']:.1f}, "
            f"Limit-Anteil {row['truncation_rate']:.0%}",
            flush=True,
        )

    if workers == 1:
        for job in jobs:
            rows.append(evaluate_run(job))
            report(rows[-1])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(evaluate_run, j) for j in jobs]):
                rows.append(future.result())
                report(rows[-1])
    # completion order depends on the pool; the files shouldn't
    rows.sort(key=lambda r: (str(r["variant"]), int(r["seed"])))  # type: ignore

    if with_baseline:
        first = loader.load(Path(str(jobs[0]["run_dir"])) / checkpoint)
        rows.append(
            random_baseline(
                first,
                episodes,
                seed,
                max_episode_steps or first.config.eval_max_episode_steps,
            )
        )
    return rows, aggregate(rows, tuple(AGGREGATED) + loader.metrics)


def print_table(aggregated: Sequence[Dict[str, object]]) -> None:
    """One line per variant with the numbers a reader looks at first."""
    # plain ASCII throughout: the Windows console is cp1252 and turns
    # characters like the plus-minus sign into replacement glyphs
    print(
        f"\n{'Variante':22s} {'n':>3} {'Median':>8} {'Bereich':>15} {'p90':>7} "
        f"{'->10':>8} {'erreicht':>9} {'Flap':>6} {'Limit':>6}"
    )
    for entry in aggregated:
        low = entry.get("median_score_min")
        high = entry.get("median_score_max")
        span = f"{low:.0f} - {high:.0f}" if low is not None else ""
        steps = entry.get("steps_to_10")
        p90 = entry.get("pooled_p90_score")
        print(
            f"{str(entry['variant']):22s} {entry['seeds']:>3} "
            f"{entry.get('median_score', float('nan')):>8.1f} {span:>15} "
            f"{(f'{p90:.0f}' if p90 is not None else '-'):>7} "
            f"{(f'{steps / 1e6:.2f}M' if steps else '-'):>8} "
            f"{str(entry.get('steps_to_10_reached', '')):>9} "
            f"{entry.get('mean_flap_rate', float('nan')):>6.3f} "
            f"{entry.get('truncation_rate', 0.0):>6.2f}"
        )
    print(
        "\nMedian  = Median der Lauf-Mediane, Bereich = schlechtester bis bester "
        "Seed.\np90     = 90. Perzentil ueber alle Episoden aller Seeds.\n"
        "->10    = Schritte, bis der gleitende Trainings-Score 10 hielt (Median "
        "der Seeds,\n          die es erreichten); nicht zensierbar, daher auch "
        "zwischen Verfahren vergleichbar.\n"
        "Limit   = Anteil Episoden im Frame-Limit. Groesser 0 heisst: Score nach "
        "oben\n          zensiert, mit hoeherem --max-episode-steps nachmessen.\n"
        "Ist der Bereich breiter als der Abstand zweier Varianten, sagt die "
        "Tabelle allein\nnichts -- dafuer significance.py."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure and aggregate a study.")
    parser.add_argument("study_root", type=Path)
    parser.add_argument(
        "--algorithm",
        required=True,
        help=f"one of {sorted(LOADERS)}, or 'package.module:RUN_LOADER'",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument(
        "--checkpoint", default=None, help="file name; default: the method's own"
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="frame limit; defaults to each run's eval_max_episode_steps",
    )
    parser.add_argument("--workers", type=int, default=0, help="0 = cores - 1")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--variants", nargs="+", default=None, help="only measure these variants"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    loader = resolve_loader(args.algorithm)
    checkpoint = args.checkpoint or loader.default_checkpoint
    rows, aggregated = evaluate_study(
        args.study_root,
        loader,
        episodes=args.episodes,
        seed=args.seed,
        checkpoint=checkpoint,
        max_episode_steps=args.max_episode_steps,
        workers=args.workers,
        with_baseline=not args.no_baseline,
        variants=args.variants,
    )
    runs_csv, aggregate_csv = output_paths(
        args.study_root,
        checkpoint,
        loader.default_checkpoint,
        args.max_episode_steps,
        args.variants,
    )
    write_csv(runs_csv, rows)
    write_csv(aggregate_csv, aggregated)
    print_table(aggregated)
    print(f"\ngeschrieben: {runs_csv}\n             {aggregate_csv}")


if __name__ == "__main__":
    main()
