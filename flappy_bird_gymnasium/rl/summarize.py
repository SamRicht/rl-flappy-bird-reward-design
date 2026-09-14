"""Evaluates every run of a study and aggregates the results per variant.

Training curves say how fast a configuration learned; this module says what its
final policy actually does.  Every run is evaluated on the *same* seeds, so all
variants see identical pipe layouts and the comparison is paired.

Usage::

    python -m flappy_bird_gymnasium.rl.summarize runs/reward_study \
        --episodes 50 --out runs/reward_study/evaluations.json
"""

import argparse
import csv
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from flappy_bird_gymnasium.rl.evaluate import evaluate, load_run
from flappy_bird_gymnasium.rl.plotting import read_runs
from flappy_bird_gymnasium.rl.train import limit_torch_threads

#: Score thresholds for the sample-efficiency metric.
THRESHOLDS = (10, 25, 50, 100, 200)


def steps_to_threshold(
    run_dir: Path,
    thresholds: tuple = THRESHOLDS,
    window: int = 20,
) -> Dict[str, Optional[int]]:
    """Environment steps until the policy first sustains a score threshold.

    Unlike the final score, this metric cannot be censored by the episode step
    limit, so it still separates variants once they all start saturating.  A
    threshold counts as reached when the *rolling mean* over ``window`` episodes
    crosses it, which keeps a single lucky episode from triggering it.

    Args:
        run_dir: A finished run directory containing ``episodes.csv``.
        thresholds: Score levels to look for.
        window: Number of consecutive episodes averaged before comparing.

    Returns:
        Mapping ``"steps_to_<threshold>"`` to the step count, or `None` when the
        run never reached that threshold.
    """
    steps: List[int] = []
    scores: List[float] = []
    with (run_dir / "episodes.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            steps.append(int(float(row["global_step"])))
            scores.append(float(row["score"]))

    result: Dict[str, Optional[int]] = {f"steps_to_{t}": None for t in thresholds}
    if len(scores) < window:
        return result

    values = np.asarray(scores, dtype=float)
    kernel = np.ones(window) / window
    rolling = np.convolve(values, kernel, mode="valid")
    # rolling[i] covers episodes i .. i+window-1, credited to the last of them.
    rolling_steps = np.asarray(steps[window - 1 :])

    for threshold in thresholds:
        hit = np.flatnonzero(rolling >= threshold)
        if hit.size:
            result[f"steps_to_{threshold}"] = int(rolling_steps[hit[0]])
    return result


#: Metrics aggregated across the seeds of a variant.
AGGREGATED = [
    "score_mean",
    "score_median",
    "score_max",
    "length_mean",
    "flap_rate_mean",
    "gap_offset_mean",
    "truncation_rate",
    "return_mean",
]


def _evaluate_run(job: Dict[str, object]) -> Dict[str, object]:
    """Worker: evaluates one run directory and tags the result with its name.

    A strong policy plays for thousands of frames, so evaluating a whole study
    serially takes longer than training it.  This runs in a process pool, with
    one Torch thread each.
    """
    limit_torch_threads(1)
    run_dir = Path(job["run_dir"])
    model, _, reward_config, env_config = load_run(run_dir, job["checkpoint"])
    result = evaluate(
        model,
        env_config,
        reward_config,
        episodes=job["episodes"],
        seed=job["seed"],
        deterministic=job["deterministic"],
        max_steps=job["max_steps"],
    )
    result["run"] = run_dir.name
    result["variant"] = job["variant"]
    return result


def summarize_study(
    runs_root: Path,
    episodes: int = 50,
    seed: int = 10_000,
    checkpoint: str = "model.pt",
    deterministic: bool = False,
    max_steps: Optional[int] = None,
    workers: int = 1,
    only: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Evaluates every run and aggregates per variant.

    Args:
        runs_root: Directory holding the study's run directories.
        episodes: Evaluation episodes per run.
        seed: Base seed, shared by every run so the comparison is paired.
        checkpoint: Which checkpoint to load from each run.
        deterministic: Use the arg-max action instead of sampling.
        max_steps: Episode step limit for the evaluation.  Deliberately separate
            from the training limit: a limit that a good policy reaches censors
            the score, and the censoring hits the best variants hardest.
        workers: Number of runs evaluated concurrently.
        only: Restrict the evaluation to these variants.  Useful for re-measuring
            just the censored ones at a higher limit -- note that results
            obtained under different limits are not directly comparable and the
            limit must be reported alongside them.

    Returns:
        Mapping from variant to its aggregated metrics, including the mean over
        seeds, the spread between seeds, and the per-seed values.
    """
    variants = read_runs(runs_root)
    if only:
        unknown = set(only) - set(variants)
        if unknown:
            raise KeyError(
                f"unknown variants {sorted(unknown)}; " f"available: {sorted(variants)}"
            )
        variants = {k: v for k, v in variants.items() if k in only}
    jobs = [
        {
            "run_dir": str(run_dir),
            "variant": variant,
            "checkpoint": checkpoint,
            "episodes": episodes,
            "seed": seed,
            "deterministic": deterministic,
            "max_steps": max_steps,
        }
        for variant, run_dirs in variants.items()
        for run_dir in run_dirs
        if (run_dir / checkpoint).exists()
    ]
    print(f"evaluating {len(jobs)} runs on {workers} workers")

    by_variant: Dict[str, List[Dict[str, object]]] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_evaluate_run, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                by_variant.setdefault(result["variant"], []).append(result)
                print(
                    f"  [{done}/{len(jobs)}] {result['run']}: "
                    f"score {result['score_mean']:.2f} "
                    f"(median {result['score_median']:.0f}, "
                    f"max {result['score_max']}, "
                    f"trunc {result['truncation_rate']:.0%})",
                    flush=True,
                )
    else:
        limit_torch_threads(1)
        for done, job in enumerate(jobs, start=1):
            result = _evaluate_run(job)
            by_variant.setdefault(result["variant"], []).append(result)
            print(
                f"  [{done}/{len(jobs)}] {result['run']}: "
                f"score {result['score_mean']:.2f}",
                flush=True,
            )

    results: Dict[str, Dict[str, object]] = {}
    for variant in variants:
        per_seed = sorted(by_variant.get(variant, []), key=lambda r: r["run"])
        if not per_seed:
            continue

        aggregated: Dict[str, object] = {"n_seeds": len(per_seed)}
        for key in AGGREGATED:
            values = [float(r[key]) for r in per_seed]
            aggregated[key] = float(np.mean(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))
        # Pooled scores across all seeds, for a distribution-level view.
        pooled = [s for r in per_seed for s in r["scores"]]
        aggregated["pooled_scores"] = pooled
        aggregated["pooled_median"] = float(np.median(pooled))
        aggregated["pooled_p90"] = float(np.percentile(pooled, 90))
        aggregated["per_seed"] = [
            {"run": r["run"], "score_mean": r["score_mean"]} for r in per_seed
        ]

        # Sample efficiency, read back from each run's episode log.  Runs that
        # never reached a threshold are reported rather than silently dropped,
        # because "4 of 8 seeds never got there" is itself the result.
        for threshold in THRESHOLDS:
            key = f"steps_to_{threshold}"
            reached = []
            for entry in aggregated["per_seed"]:
                value = steps_to_threshold(runs_root / entry["run"])[key]
                if value is not None:
                    reached.append(value)
            aggregated[key] = float(np.median(reached)) if reached else None
            aggregated[f"{key}_n_reached"] = len(reached)

        results[variant] = aggregated

    return results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode step limit for the evaluation; keep it well above what "
        "the best policy reaches, otherwise the scores are censored.",
    )
    parser.add_argument("--workers", type=int, default=0, help="0 = cores - 1")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Only evaluate these variants, e.g. for re-measuring the "
        "censored ones at a higher --max-steps.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    workers = args.workers or max(1, (multiprocessing.cpu_count() // 2) - 1)
    results = summarize_study(
        args.runs_root,
        episodes=args.episodes,
        seed=args.seed,
        checkpoint=args.checkpoint,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        workers=workers,
        only=args.variants,
    )

    order = sorted(results, key=lambda k: -results[k]["score_mean"])
    print()
    print(
        f"{'variant':<14}{'score':>8}{'spread':>16}{'median':>8}{'p90':>7}"
        f"{'trunc':>7}{'flaps':>8}{'gapoff':>9}{'len':>8}"
    )
    for variant in order:
        row = results[variant]
        spread = f"{row['score_mean_min']:.1f}-{row['score_mean_max']:.1f}"
        print(
            f"{variant:<14}{row['score_mean']:>8.2f}{spread:>16}"
            f"{row['pooled_median']:>8.0f}{row['pooled_p90']:>7.0f}"
            f"{row['truncation_rate']:>6.0%}{row['flap_rate_mean']:>9.3f}"
            f"{row['gap_offset_mean']:>9.4f}{row['length_mean']:>8.0f}"
        )

    out = args.out or args.runs_root / "evaluations.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
