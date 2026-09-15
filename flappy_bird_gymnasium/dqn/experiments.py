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
``reward``
    Every reward preset, repeated over several seeds.

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


def run_grid(
    specs: List[Dict[str, object]],
    runs_root: Path,
    workers: int,
) -> List[Dict[str, object]]:
    """Executes the job specifications in a process pool.

    Returns:
        The summaries of all runs, in completion order.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "study.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")

    print(f"{len(specs)} Laeufe auf {workers} Prozessen -> {runs_root}", flush=True)
    started = time.time()
    summaries: List[Dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, spec): spec for spec in specs}
        for done, future in enumerate(as_completed(futures), start=1):
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


def base_config(args: argparse.Namespace) -> DQNConfig:
    """The configuration shared by every run of a study."""
    return DQNConfig.from_dict(vars(args))


def study_seeds(args: argparse.Namespace) -> List[tuple]:
    """One configuration repeated over seeds, to measure the spread."""
    config = base_config(args)
    return [("baseline", replace(config, seed=seed)) for seed in range(args.seeds)]


def study_ablation(args: argparse.Namespace) -> List[tuple]:
    """Each algorithmic addition switched off in turn, over several seeds."""
    config = base_config(args)
    names = args.variants or list(ABLATIONS)
    return [
        (name, replace(config, seed=seed, **ABLATIONS[name]))
        for name in names
        for seed in range(args.seeds)
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
        for seed in range(args.seeds)
    ]


def study_reward(args: argparse.Namespace) -> List[tuple]:
    """Every (or the selected) reward preset, over several seeds."""
    config = base_config(args)
    names = args.rewards or sorted(PRESETS)
    return [
        (name, replace(config, seed=seed, reward_preset=name))
        for name in names
        for seed in range(args.seeds)
    ]


STUDIES = {
    "seeds": study_seeds,
    "ablation": study_ablation,
    "sweep": study_sweep,
    "reward": study_reward,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a grid of DQN trainings.")
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument("--seeds", type=int, default=3, help="seeds per variant")
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
        "--variants", nargs="+", choices=sorted(ABLATIONS), help="ablation: which ones"
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
