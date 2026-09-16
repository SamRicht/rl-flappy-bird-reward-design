"""Runs a grid of DQN trainings in parallel and collects their summaries.

A single DQN run is noisy: the dqn_v2 baseline swung between an evaluation
score of 1.4 and 79.0 within 50.000 steps. Any statement of the form "variant A
beats variant B" therefore needs several seeds per variant, or it is a
statement about luck.

Four studies are predefined:

``seeds``
    One configuration over N seeds. Measures how much spread the algorithm
    itself produces -- the yardstick every other comparison is read against.
``ablation``
    Double DQN, the dueling head and n-step returns switched off one at a time,
    plus a textbook DQN with all three off. Shows what each part contributes.
``sweep``
    One hyperparameter varied over a list of values, everything else fixed.
``params``
    Several hyperparameters at once, each varied on its own around a shared
    baseline -- the search for a better setting before the final run.
``reward``
    Every reward preset, repeated over several seeds.
``reward_terms``
    The alive bonus and the ceiling penalty switched off one at a time and
    together -- which single term a difference between schemes comes from.

Each run keeps one core busy, so runs go into a process pool with Torch pinned
to a single thread per worker -- otherwise the runs fight over cores and each
one ends up slower than it would be alone.

Examples::

    python -m flappy_bird_gymnasium.dqn.experiments seeds --seeds 8 \
        --total-steps 1000000
    python -m flappy_bird_gymnasium.dqn.experiments ablation --seeds 3
    python -m flappy_bird_gymnasium.dqn.experiments sweep --param learning_rate \
        --values 5e-5 1e-4 3e-4 --seeds 3
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from flappy_bird_gymnasium.dqn.config import DQNConfig, add_config_arguments
from flappy_bird_gymnasium.dqn.train import train
from flappy_bird_gymnasium.rl.rewards import PRESETS

def _parse_hidden(value: str) -> tuple:
    """Parses ``"256x256"`` (or ``"256,256"``) into ``(256, 256)``."""
    parts = str(value).replace(",", "x").split("x")
    return tuple(int(part) for part in parts if part)


def _label_value(value: object) -> str:
    """A filesystem-safe fragment for a variant directory name."""
    if isinstance(value, tuple):
        return "x".join(str(v) for v in value)
    return str(value).replace(".", "p").replace("-", "m")


#: Hyperparameters the ``sweep`` study may vary, with the parser for their
#: command-line values.
SWEEPABLE: Dict[str, Callable] = {
    "learning_rate": float,
    "gamma": float,
    "n_step": int,
    "batch_size": int,
    "buffer_size": int,
    "target_update_interval": int,
    "epsilon_decay_steps": int,
    "learning_starts": int,
    "train_freq": int,
    "pipe_gap": int,
    "hidden": _parse_hidden,
}

#: The ablation variants: what is switched off relative to the full agent.
ABLATIONS: Dict[str, Dict[str, object]] = {
    "full": {},
    "no_double": {"double_dqn": False},
    "no_dueling": {"dueling": False},
    "no_nstep": {"n_step": 1},
    # everything off at once -- DQN as it was originally published
    "vanilla": {"double_dqn": False, "dueling": False, "n_step": 1},
}


#: The ``params`` study: the values tried per hyperparameter, one factor at a
#: time. Only values *other* than the default are listed -- the baseline runs
#: once for all of them, so N parameters cost N x (values - 1) + 1 arms instead
#: of N x values. The values bracket the default from both sides, which is what
#: tells "the default is a peak" apart from "we never looked further".
PARAM_GRID: Dict[str, List] = {
    # the classic first knob; 3e-4 is expected to be unstable at large Q-values
    "learning_rate": [5e-5, 3e-4],
    # sets how far ahead the agent plans; with the dense alive reward it also
    # sets how large the Q-values get: 0.1 / (1 - gamma) is 2, 10 or 100
    "gamma": [0.95, 0.999],
    # the ablation found n=3 well ahead of n=1 under `legacy`; a faster reward
    # scheme may shift that, and n=5 was never tried at this budget
    "n_step": [1, 5],
    # too frequent and the target chases itself, too rare and it is stale
    "target_update_interval": [250, 4000],
    # a faster-learning reward may no longer need 200k steps of exploration
    "epsilon_decay_steps": [100_000, 400_000],
    # 12 inputs may not need 256x256; a smaller net would also train faster
    "hidden": [(64, 64), (512, 512)],
}

#: The ``reward_terms`` study: a 2x2 over the two terms that separate the
#: winning schemes of the reward study from the losing ones. Everything runs in
#: "additive" mode so the composition mode is not a second difference, and the
#: pipe bonus and death penalty stay on everywhere -- only the two dense terms
#: vary. ``both`` reproduces the ``additive`` preset, ``neither`` the ``sparse``
#: one, but as part of one factorial design rather than two unrelated presets.
REWARD_TERMS: Dict[str, Dict[str, float]] = {
    "both": {},
    "no_ceiling": {"ceiling": 0.0},
    "no_alive": {"alive": 0.0},
    "neither": {"alive": 0.0, "ceiling": 0.0},
}


def _run_one(spec: Dict[str, object]) -> Dict[str, object]:
    """Worker entry point: trains one configuration and returns its summary.

    Takes a plain dictionary rather than a dataclass so the payload pickles
    cleanly across the process boundary.
    """
    config = DQNConfig.from_dict(spec["config"])
    summary = train(
        config,
        Path(spec["run_dir"]),
        verbose=False,
        torch_threads=1,
    )
    summary["label"] = spec["label"]
    summary["variant"] = spec["variant"]
    return summary


def build_specs(
    variants: Sequence[tuple],
    runs_root: Path,
) -> List[Dict[str, object]]:
    """Turns ``(variant, config)`` pairs into picklable job specifications."""
    specs = []
    for variant, config in variants:
        label = f"{variant}_seed{config.seed}"
        specs.append(
            {
                "label": label,
                "variant": variant,
                "run_dir": str(runs_root / label),
                "config": replace(config, run_name=label).to_dict(),
            }
        )
    return specs


def completed_summary(spec: Dict[str, object]) -> Optional[Dict[str, object]]:
    """The summary of a run that already finished with this exact config.

    Lets an interrupted study be restarted with the same command: finished runs
    are kept, unfinished ones start over from scratch. Resuming a run midway is
    deliberately not supported -- checkpoints hold no replay buffer and no RNG
    state, so a resumed run would not match one that ran through.
    """
    run_dir = Path(spec["run_dir"])
    summary_path, config_path = run_dir / "summary.json", run_dir / "config.json"
    if not (summary_path.exists() and config_path.exists()):
        return None
    stored = json.loads(config_path.read_text(encoding="utf-8"))
    if stored != json.loads(json.dumps(spec["config"])):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["label"] = spec["label"]
    summary["variant"] = spec["variant"]
    return summary


def run_grid(
    specs: List[Dict[str, object]],
    runs_root: Path,
    workers: int,
) -> List[Dict[str, object]]:
    """Executes the job specifications in a process pool.

    Runs that already finished with the same config are skipped, see
    `completed_summary`.

    Returns:
        The summaries of all runs, in completion order.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "study.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")

    summaries: List[Dict[str, object]] = []
    pending = []
    for spec in specs:
        summary = completed_summary(spec)
        if summary is None:
            pending.append(spec)
        else:
            summaries.append(summary)
    if summaries:
        print(
            f"{len(summaries)} Laeufe bereits fertig, uebersprungen: "
            + ", ".join(s["label"] for s in summaries),
            flush=True,
        )

    print(f"{len(pending)} Laeufe auf {workers} Prozessen -> {runs_root}", flush=True)
    started = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, spec): spec for spec in pending}
        for done, future in enumerate(as_completed(futures), start=len(summaries) + 1):
            summary = future.result()
            summaries.append(summary)
            censored = (
                " [zensiert]" if summary.get("best_eval_truncation_rate", 0) >= 1 else ""
            )
            print(
                f"  [{done}/{len(specs)}] {summary['label']}: "
                f"best eval {summary['best_eval_score']:.2f}{censored} "
                f"({(time.time() - started) / 60:.1f} min)",
                flush=True,
            )

    (runs_root / "summaries.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(f"fertig in {(time.time() - started) / 60:.1f} min", flush=True)
    return summaries


def seed_range(args: argparse.Namespace) -> range:
    """The seeds of a study: ``--seeds`` of them, starting at ``--seed-offset``.

    A final run of a configuration that was *chosen* on seeds 0 .. n-1 must be
    measured on fresh seeds, or the choice has adapted to the luck of exactly
    those seeds and the reported number comes out too optimistic.
    """
    return range(args.seed_offset, args.seed_offset + args.seeds)


def base_config(args: argparse.Namespace) -> DQNConfig:
    """The configuration shared by every run of a study."""
    return DQNConfig.from_dict(vars(args))


def study_seeds(args: argparse.Namespace) -> List[tuple]:
    """One configuration repeated over seeds, to measure the spread."""
    config = base_config(args)
    return [("baseline", replace(config, seed=seed)) for seed in seed_range(args)]


def study_ablation(args: argparse.Namespace) -> List[tuple]:
    """Each algorithmic addition switched off in turn, over several seeds."""
    config = base_config(args)
    names = args.variants or list(ABLATIONS)
    return [
        (name, replace(config, seed=seed, **ABLATIONS[name]))
        for name in names
        for seed in seed_range(args)
    ]


def study_sweep(args: argparse.Namespace) -> List[tuple]:
    """One hyperparameter varied over the given values, over several seeds."""
    config = base_config(args)
    cast = SWEEPABLE[args.param]
    return [
        (
            f"{args.param}_{_label_value(cast(value))}",
            replace(config, seed=seed, **{args.param: cast(value)}),
        )
        for value in args.values
        for seed in seed_range(args)
    ]


def study_params(args: argparse.Namespace) -> List[tuple]:
    """Several hyperparameters, each varied on its own around the baseline.

    One factor at a time: every variant changes exactly one parameter, so a
    difference can be attributed to it. The baseline runs once and serves as the
    reference for all of them -- the default value of each parameter is
    therefore *not* repeated per parameter, which is what keeps 6 parameters
    affordable.

    What this design cannot find is an interaction: two changes that only pay
    off together. Finding those needs a grid, which costs the product instead of
    the sum of the arms.
    """
    config = base_config(args)
    names = args.params or list(PARAM_GRID)
    variants = [("baseline", replace(config, seed=seed)) for seed in seed_range(args)]
    variants += [
        (
            f"{param}_{_label_value(value)}",
            replace(config, seed=seed, **{param: value}),
        )
        for param in names
        for value in PARAM_GRID[param]
        for seed in seed_range(args)
    ]
    return variants


def study_reward(args: argparse.Namespace) -> List[tuple]:
    """Every (or the selected) reward preset, over several seeds."""
    config = base_config(args)
    names = args.rewards or sorted(PRESETS)
    return [
        (name, replace(config, seed=seed, reward_preset=name))
        for name in names
        for seed in seed_range(args)
    ]


def study_reward_terms(args: argparse.Namespace) -> List[tuple]:
    """The alive bonus and the ceiling penalty switched on and off, 2x2.

    The reward study compares whole schemes; this one asks which single term
    the difference comes from.
    """
    config = replace(base_config(args), reward_preset="additive")
    names = args.variants or list(REWARD_TERMS)
    return [
        (name, replace(config, seed=seed, reward_overrides=dict(REWARD_TERMS[name])))
        for name in names
        for seed in seed_range(args)
    ]


STUDIES = {
    "seeds": study_seeds,
    "ablation": study_ablation,
    "sweep": study_sweep,
    "params": study_params,
    "reward": study_reward,
    "reward_terms": study_reward_terms,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a grid of DQN trainings.")
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument("--seeds", type=int, default=3, help="seeds per variant")
    parser.add_argument(
        "--seed-offset", type=int, default=0, help="first seed; use fresh seeds for final runs"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="0 = number of cores minus two"
    )
    parser.add_argument("--runs-root", type=Path, default=None)

    # the same flags as train.py, from the same definition, so the two entry
    # points cannot disagree about names or defaults
    add_config_arguments(parser)
    parser.set_defaults(total_steps=1_000_000)

    # study-specific
    parser.add_argument("--param", choices=sorted(SWEEPABLE), help="sweep: what to vary")
    parser.add_argument("--values", nargs="+", help="sweep: the values to try")
    parser.add_argument(
        "--params",
        nargs="+",
        choices=sorted(PARAM_GRID),
        help="params: which hyperparameters (default: all of them)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(set(ABLATIONS) | set(REWARD_TERMS)),
        help="ablation / reward_terms: which variants",
    )
    parser.add_argument(
        "--rewards", nargs="+", choices=sorted(PRESETS), help="reward: which presets"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.study == "sweep" and not (args.param and args.values):
        raise SystemExit("sweep braucht --param und --values")

    variants = STUDIES[args.study](args)
    runs_root = args.runs_root or Path("runs") / f"study_{args.study}"
    specs = build_specs(variants, runs_root)

    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    workers = min(workers, len(specs))
    run_grid(specs, runs_root, workers)

    print(
        f"\nAuswertung:\n"
        f"  python -m flappy_bird_gymnasium.dqn.summarize {runs_root} --episodes 30"
    )


if __name__ == "__main__":
    main()
