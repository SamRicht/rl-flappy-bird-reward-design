"""Evaluates every run of a study and aggregates the results per variant.

Training curves say how *fast* a configuration learned; this module says what
its final policy actually does, and how much of the difference between two
variants is real rather than seed luck.

Three rules make the comparison honest:

*Paired evaluation.* Every run is measured on the same evaluation seeds, so all
variants face identical pipe layouts. Without this, a variant can win by having
drawn easier levels.

*Uncensored measurement.* The frame limit used here comes from the run's
``eval_max_episode_steps``, far above the training limit. Measuring against the
training limit collapses every competent variant onto the same score -- which
is exactly what happened in the dqn_v2 baseline, where a policy worth 304 pipes
reported 79.

*Sample efficiency alongside final score.* ``steps_to_<threshold>`` (taken from
each run's ``summary.json``) is the number of environment steps until the
rolling mean score first reached a level. It cannot be censored by any limit,
so it still separates variants after their final scores saturate -- and it is
comparable across algorithms, which matters for the PPO / DQN / Q-Learning /
CNN comparison.

Usage::

    python -m flappy_bird_gymnasium.dqn.summarize runs/study_ablation --episodes 30
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from flappy_bird_gymnasium.dqn.config import DQNConfig
from flappy_bird_gymnasium.dqn.env_utils import make_env, rollout, summarize_rollout
from flappy_bird_gymnasium.dqn.evaluate import play
from flappy_bird_gymnasium.dqn.train import THRESHOLDS

#: Metrics carried through from the per-run evaluation.
EVAL_METRICS = [
    "mean_score",
    "median_score",
    "max_score",
    "min_score",
    "mean_length",
    "mean_flap_rate",
    "truncation_rate",
]

#: Sample-efficiency metrics read from each run's summary.json, derived from
#: the same thresholds the training run used so the two cannot drift apart.
EFFICIENCY_METRICS = [f"steps_to_{threshold}" for threshold in THRESHOLDS]

#: Columns of the two output files.
PER_RUN_FIELDS = [
    "algorithm",
    "run",
    "variant",
    "run_seed",
    *EVAL_METRICS,
    *EFFICIENCY_METRICS,
    "minutes",
]
AGGREGATE_FIELDS = [
    "algorithm",
    "variant",
    "seeds",
    "score_mean",
    "score_std",
    "score_min_seed",
    "score_max_seed",
    "length_mean",
    "flap_rate_mean",
    "truncation_rate",
    "minutes_mean",
    *EFFICIENCY_METRICS,
    *[f"{metric}_reached" for metric in EFFICIENCY_METRICS],
]


def _evaluate_one(job: Dict) -> Dict:
    """Worker: evaluates one run directory and tags the result."""
    from flappy_bird_gymnasium.dqn.env_utils import limit_torch_threads

    limit_torch_threads(1)
    run_dir = Path(job["run_dir"])
    result = play(
        checkpoint=run_dir / job["checkpoint"],
        episodes=job["episodes"],
        render=False,
        max_episode_steps=job.get("max_episode_steps"),
        seed=job["seed"],
        verbose=False,
    )
    result["run"] = run_dir.name
    result["variant"] = job["variant"]
    result["run_seed"] = job["run_seed"]
    # tagged so these rows can be concatenated with the PPO, Q-Learning and
    # CNN results into one comparison table
    result["algorithm"] = "dqn"
    return result


def random_baseline(
    config: DQNConfig, episodes: int, seed: int, max_episode_steps: int
) -> Dict:
    """Measures a uniformly random policy on the same episodes as the agents.

    Every score in a comparison is read against this. Without it "DQN reaches
    304 pipes" carries no information -- and across four learning methods the
    random floor is the one row every chart can be anchored to. It goes through
    the same `rollout` as the trained agents, so it is measured identically by
    construction rather than by careful copying.
    """
    env = make_env(config, render_mode=None, max_episode_steps=max_episode_steps)
    rng = np.random.default_rng(seed)
    measured = rollout(env, lambda obs: int(rng.integers(2)), episodes, seed)
    env.close()
    return {
        "run": "random",
        "variant": "random",
        "run_seed": seed,
        "algorithm": "random",
        **summarize_rollout(measured),
    }


def collect_runs(study_root: Path) -> List[Dict]:
    """Finds every finished run under `study_root`.

    A run counts as finished when it has both a `config.json` and a checkpoint.
    """
    runs = []
    for config_path in sorted(study_root.glob("*/config.json")):
        run_dir = config_path.parent
        if not (run_dir / "best.pt").exists():
            print(f"  uebersprungen (kein best.pt): {run_dir.name}")
            continue
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        # "variant_seedN" -> "variant"
        summary_path = run_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
        # "variant_seedN" -> "variant"
        variant = run_dir.name.rsplit("_seed", 1)[0]
        runs.append(
            {
                "run_dir": run_dir,
                "variant": variant,
                "run_seed": config.get("seed"),
                "config": config,
                "summary": summary,
            }
        )
    return runs


def evaluate_study(
    runs: List[Dict],
    episodes: int,
    seed: int,
    checkpoint: str,
    workers: int,
    max_episode_steps: Optional[int] = None,
) -> List[Dict]:
    """Evaluates every run on identical episode seeds, so results are paired."""
    jobs = [
        {
            "run_dir": str(run["run_dir"]),
            "variant": run["variant"],
            "run_seed": run["run_seed"],
            "episodes": episodes,
            "seed": seed,
            "checkpoint": checkpoint,
            "max_episode_steps": max_episode_steps,
        }
        for run in runs
    ]

    print(f"{len(jobs)} Laeufe x {episodes} Episoden auf {workers} Prozessen ...")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_evaluate_one, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"  [{done}/{len(jobs)}] {result['run']}: "
                f"score {result['mean_score']:.1f}",
                flush=True,
            )

    # merge the sample-efficiency numbers from training
    by_name = {run["run_dir"].name: run for run in runs}
    for result in results:
        summary = by_name[result["run"]]["summary"]
        for metric in EFFICIENCY_METRICS:
            result[metric] = summary.get(metric)
        result["minutes"] = summary.get("minutes")
    return results


def aggregate(results: List[Dict]) -> List[Dict]:
    """Averages the per-run results over the seeds of each variant."""
    variants: Dict[str, List[Dict]] = {}
    for result in results:
        variants.setdefault(result["variant"], []).append(result)

    rows = []
    for variant, runs in sorted(variants.items()):
        scores = np.array([r["mean_score"] for r in runs], dtype=float)
        row: Dict[str, object] = {
            "algorithm": runs[0].get("algorithm", "dqn"),
            "variant": variant,
            "seeds": len(runs),
            "score_mean": float(scores.mean()),
            # spread *across seeds* -- the yardstick for whether a difference
            # between two variants means anything
            "score_std": float(scores.std(ddof=1)) if len(runs) > 1 else 0.0,
            "score_min_seed": float(scores.min()),
            "score_max_seed": float(scores.max()),
            "length_mean": float(np.mean([r["mean_length"] for r in runs])),
            "flap_rate_mean": float(np.mean([r["mean_flap_rate"] for r in runs])),
            "truncation_rate": float(np.mean([r["truncation_rate"] for r in runs])),
            "minutes_mean": float(
                np.mean([r["minutes"] for r in runs if r.get("minutes") is not None])
                if any(r.get("minutes") is not None for r in runs)
                else 0.0
            ),
        }
        for metric in EFFICIENCY_METRICS:
            reached = [r[metric] for r in runs if r.get(metric) is not None]
            # median, because a variant that never reached the level has no
            # number at all and a mean over the rest would flatter it
            row[metric] = float(np.median(reached)) if reached else None
            row[f"{metric}_reached"] = f"{len(reached)}/{len(runs)}"
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[Dict]) -> None:
    """Prints the aggregate table, best variant first."""
    rows = sorted(rows, key=lambda r: r["score_mean"], reverse=True)
    # plain ASCII throughout: the Windows console is cp1252 and turns
    # characters like the plus-minus sign into replacement glyphs
    header = (
        f"{'Variante':<16}{'Seeds':>6}{'Score':>18}{'Spanne':>16}"
        f"{'bis 10':>10}{'bis 25':>10}{'Trunc':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        span = f"{row['score_min_seed']:.0f} - {row['score_max_seed']:.0f}"
        to10 = f"{row['steps_to_10'] / 1000:.0f}k" if row.get("steps_to_10") else "-"
        to25 = f"{row['steps_to_25'] / 1000:.0f}k" if row.get("steps_to_25") else "-"
        score = f"{row['score_mean']:.1f} +/- {row['score_std']:.1f}"
        print(
            f"{row['variant']:<16}{row['seeds']:>6}{score:>18}"
            f"{span:>16}{to10:>10}{to25:>10}{row['truncation_rate']:>8.2f}"
        )
    print(
        "\nScore = mittlere passierte Roehren, gemittelt ueber die Seeds;\n"
        "        +/- ist die Streuung ZWISCHEN den Seeds -- der Massstab dafuer,\n"
        "        ob ein Unterschied zwischen zwei Varianten etwas bedeutet.\n"
        "Spanne = schlechtester bis bester Seed.\n"
        "bis 10 / bis 25 = Umgebungsschritte, bis der gleitende Score das Niveau\n"
        "        hielt (Median ueber die Seeds). Nicht zensierbar, daher auch\n"
        "        zwischen Algorithmen vergleichbar.\n"
        "Trunc  = Anteil Episoden im Frame-Limit. Groesser 0 heisst: Score nach\n"
        "        oben zensiert, mit hoeherem --max-episode-steps nachmessen."
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate a DQN study.")
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--seed", type=int, default=90_000, help="evaluation seed, shared by all runs"
    )
    parser.add_argument("--checkpoint", default="best.pt", choices=["best.pt", "latest.pt"])
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="frame limit; default is each run's eval_max_episode_steps",
    )
    parser.add_argument("--workers", type=int, default=0, help="0 = cores minus two")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the random-policy reference row",
    )
    args = parser.parse_args(argv)

    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    runs = collect_runs(args.study_root)
    if not runs:
        raise SystemExit(f"keine fertigen Laeufe unter {args.study_root}")
    results = evaluate_study(
        runs,
        episodes=args.episodes,
        seed=args.seed,
        checkpoint=args.checkpoint,
        workers=workers,
        max_episode_steps=args.max_episode_steps,
    )
    rows = aggregate(results)

    if not args.no_baseline:
        print("Zufalls-Referenz ...")
        config = DQNConfig.from_dict(runs[0]["config"])
        results.append(
            random_baseline(
                config,
                args.episodes,
                args.seed,
                args.max_episode_steps or config.eval_max_episode_steps,
            )
        )

    write_csv(args.study_root / "evaluations.csv", results, PER_RUN_FIELDS)
    write_csv(args.study_root / "aggregate.csv", rows, AGGREGATE_FIELDS)

    print_table(rows)
    print(
        f"\ngeschrieben: {args.study_root / 'evaluations.csv'} (pro Lauf), "
        f"{args.study_root / 'aggregate.csv'} (pro Variante)"
    )


if __name__ == "__main__":
    main()
