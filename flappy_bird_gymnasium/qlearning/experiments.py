"""Multi-seed studies: many runs that differ in one thing, run in parallel.

A single run says almost nothing here. The score distribution has a long right
tail, so two runs of the same config can differ by a factor of ten (four seeds
of one setting gave mean scores of 51, 60, 49 and 455). So every claim needs
several seeds, and every study varies one axis while holding the rest fixed.

Usage::

    python -m flappy_bird_gymnasium.qlearning.experiments reward --seeds 5
    python -m flappy_bird_gymnasium.qlearning.experiments discretizer --seeds 5
    python -m flappy_bird_gymnasium.qlearning.experiments sweep \\
        --param epsilon_decay_steps --values 150000 300000 3000000

Each study writes ``runs/study_<name>/<variant>_seed<n>/`` plus a ``study.json``
with the job specs and a ``summaries.json`` with the run summaries. The shared
chain in :mod:`flappy_bird_gymnasium.rl.analysis` measures, tests and draws
them, ``--algorithm qlearning`` there.
"""

import argparse
import itertools
import json
import multiprocessing
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from flappy_bird_gymnasium.qlearning.config import QLearningConfig, add_config_arguments
from flappy_bird_gymnasium.qlearning.discretize import DISCRETIZERS
from flappy_bird_gymnasium.qlearning.train import train
from flappy_bird_gymnasium.rl.rewards import PRESETS


def _preset_name(value: str) -> str:
    """A reward preset, checked now rather than when the worker builds the env.

    ``QLearningConfig`` validates the discretiser and the update rule but not
    the reward scheme, so an unknown one would otherwise surface as a
    ``KeyError`` inside a pool worker -- after the grid has already started.
    """
    if value not in PRESETS:
        raise SystemExit(
            f"unknown reward preset {value!r}; available: {sorted(PRESETS)}"
        )
    return value


#: Fields that ``sweep`` and ``matrix`` can vary, with the cast applied to the
#: CLI strings.
SWEEPABLE: Dict[str, Callable[[str], object]] = {
    "reward_preset": _preset_name,
    "learning_rate": float,
    "learning_rate_end": float,
    "learning_rate_exponent": float,
    "learning_rate_mode": str,
    "gamma": float,
    "q_init": float,
    "n_step": int,
    "algo": str,
    "epsilon_end": float,
    "epsilon_decay_steps": int,
    "discretizer": str,
    "dx_bins": int,
    "dy_bins": int,
    "vel_bins": int,
    "total_steps": int,
    "pipe_gap": int,
}

#: One-factor-at-a-time grid: each value is tried instead of the default, with
#: everything else at its default. The values bracket the default so that "the
#: default is a peak" and "the default is on a slope" look different.
#:
#: ``n_step`` is missing on purpose: its default of 1 is a hard lower bound and
#: can't be bracketed. It lives in :data:`ABLATIONS` instead, as a structural
#: change of the update rather than a number to tune.
PARAM_GRID: Dict[str, Sequence[object]] = {
    "learning_rate": (0.1, 0.5),
    "learning_rate_end": (0.005, 0.1),
    "gamma": (0.95, 0.995),
    "q_init": (5.0, 20.0),
    "epsilon_decay_steps": (150_000, 1_000_000),
}

#: Variations of the update rule itself rather than of a number in it.
#:
#: ``count_alpha`` keeps the default floor of 0.02 on purpose. Lowering it to
#: 0.001, which looks like the natural thing to do for a per-entry step size,
#: turns the best ablation into one of the worst: five seeds at 10M steps,
#: median score 179 vs 37, and ``steps_to_10`` up from 1.81M to 3.96M
#: (p = 0.0159). The low floor stays as its own variant because the gap is
#: itself a result: the rarely visited entries are still learning at a point
#: where the low floor stops them.
ABLATIONS: Dict[str, Dict[str, object]] = {
    "basis": {},
    "expected_sarsa": {"algo": "expected_sarsa"},
    "count_alpha": {"learning_rate_mode": "count"},
    "count_alpha_low": {"learning_rate_mode": "count", "learning_rate_end": 0.001},
    "n_step_3": {"n_step": 3},
    "count_alpha_n3": {"learning_rate_mode": "count", "n_step": 3},
}

#: Uniform grids for the granularity study, as (dx, dy, vel) bin counts. vel_y
#: takes exactly 20 integer values, so 20 bins resolve it exactly and only dx
#: and dy are actually varied.
GRANULARITY: Dict[str, Tuple[int, int, int]] = {
    "g05k": (16, 16, 20),
    "g15k": (24, 32, 20),
    "g26k": (32, 40, 20),
    "g45k": (40, 56, 20),
    "g81k": (56, 72, 20),
}

Variant = Tuple[str, QLearningConfig]


def base_config(args: argparse.Namespace) -> QLearningConfig:
    """The config every variant of a study starts from."""
    return QLearningConfig.from_dict(vars(args))


def seed_range(args: argparse.Namespace) -> range:
    """Seeds of the study: ``seed_offset`` .. ``seed_offset + seeds - 1``."""
    return range(args.seed_offset, args.seed_offset + args.seeds)


def _variants(
    args: argparse.Namespace, overrides: Dict[str, Dict[str, object]]
) -> List[Variant]:
    """Crosses ``overrides`` with the seed range into labelled configs."""
    base = base_config(args)
    out: List[Variant] = []
    for label, changes in overrides.items():
        for seed in seed_range(args):
            out.append((label, replace(base, seed=seed, **changes)))
    return out


def study_seeds(args: argparse.Namespace) -> List[Variant]:
    """The default config on several seeds, i.e. the noise floor.

    Worth running once before any comparison: it shows how big a difference
    has to be before it means anything.
    """
    return _variants(args, {"default": {}})


def study_reward(args: argparse.Namespace) -> List[Variant]:
    """Every reward scheme in ``rl.rewards``. The project's actual question."""
    return _variants(args, {name: {"reward_preset": name} for name in sorted(PRESETS)})


def study_reward_terms(args: argparse.Namespace) -> List[Variant]:
    """A 2x2 over two single terms, on top of one preset.

    Whole presets differ in several terms at once, so a difference between
    two of them can't be attributed to any one term. This varies exactly two.
    """
    overrides: Dict[str, Dict[str, object]] = {}
    for alive in (0.0, 0.1):
        for ceiling in (-0.5, 0.0):
            label = f"alive{alive:g}_ceiling{ceiling:g}"
            overrides[label] = {
                "reward_preset": "additive",
                "reward_overrides": {"alive": alive, "ceiling": ceiling},
            }
    return _variants(args, overrides)


def study_discretizer(args: argparse.Namespace) -> List[Variant]:
    """The three registered rasterisation schemes.

    The tabular-specific axis; none of the network-based methods has this
    knob at all.
    """
    return _variants(
        args, {name: {"discretizer": name} for name in sorted(DISCRETIZERS)}
    )


def study_granularity(args: argparse.Namespace) -> List[Variant]:
    """How fine the uniform grid should be, at a fixed budget.

    Expected to be unimodal, not monotone: too coarse puts fatally different
    situations in one cell, too fine leaves too few visits per cell.
    """
    overrides = {
        label: {"discretizer": "uniform", "dx_bins": dx, "dy_bins": dy, "vel_bins": vel}
        for label, (dx, dy, vel) in GRANULARITY.items()
    }
    return _variants(args, overrides)


def study_ablation(args: argparse.Namespace) -> List[Variant]:
    """Variations of the update rule, each against the plain baseline."""
    return _variants(args, ABLATIONS)


def study_params(args: argparse.Namespace) -> List[Variant]:
    """One factor at a time over :data:`PARAM_GRID`, plus the baseline once.

    The baseline is a single variant, not one per parameter, so it isn't
    trained six times over.
    """
    overrides: Dict[str, Dict[str, object]] = {"baseline": {}}
    for param, values in PARAM_GRID.items():
        for value in values:
            overrides[
                f"{param}={value:g}" if isinstance(value, float) else f"{param}={value}"
            ] = {param: value}
    return _variants(args, overrides)


def study_sweep(args: argparse.Namespace) -> List[Variant]:
    """One field over the values given on the command line."""
    if args.param is None or not args.values:
        raise SystemExit("sweep needs --param and --values")
    if args.param not in SWEEPABLE:
        raise SystemExit(f"--param must be one of {sorted(SWEEPABLE)}")
    cast = SWEEPABLE[args.param]
    overrides = {f"{args.param}={v}": {args.param: cast(v)} for v in args.values}
    return _variants(args, overrides)


def parse_factor(text: str) -> Tuple[str, List[object]]:
    """``"q_init=2.5,10"`` into ``("q_init", [2.5, 10.0])``.

    Raises:
        SystemExit: On a malformed spec, an unknown field or an empty value
            list -- all three are typos in a command that would otherwise
            start hours of training.
    """
    if "=" not in text:
        raise SystemExit(f"--factor needs NAME=v1,v2,...; got {text!r}")
    name, raw = text.split("=", 1)
    name = name.strip()
    if name not in SWEEPABLE:
        raise SystemExit(f"unknown factor {name!r}; available: {sorted(SWEEPABLE)}")
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise SystemExit(f"factor {name!r} has no values")
    cast = SWEEPABLE[name]
    return name, [cast(v) for v in values]


def matrix_label(cell: Sequence[Tuple[str, object]]) -> str:
    """Directory label of one cell, e.g. ``reward_preset=legacy__q_init=10``.

    Spelled out rather than numbered, so a run directory says what it is
    without an index file. ``__`` separates the factors because the field
    names themselves contain single underscores, and no value may contain
    ``_seed`` -- that is what
    :func:`flappy_bird_gymnasium.rl.analysis.runs.variant_of` splits on.
    """
    return "__".join(f"{name}={value}" for name, value in cell)


def study_matrix(args: argparse.Namespace) -> List[Variant]:
    """Every combination of the factors given with ``--factor``.

    A full factorial, as opposed to the one-factor-at-a-time studies above.
    Worth the cost when factors *interact*, which here they demonstrably do:
    the best ``q_init`` depends on the reward scheme through the return that
    scheme can reach, so a one-factor sweep of either one measures the other
    as much as itself.

    Read it by pooling: a main effect averages over every level of the other
    factors, so it rests on far more than the ``--seeds`` runs of a single
    cell. Single cells are noisy at any affordable seed count and should not
    be read on their own.

    Example::

        experiments matrix --seeds 4 \\
            --factor reward_preset=legacy,sparse,shaped \\
            --factor q_init=2.5,10 \\
            --factor discretizer=uniform,adaptive
    """
    if not args.factor:
        raise SystemExit(
            "matrix needs at least one --factor NAME=v1,v2,... "
            f"(available: {sorted(SWEEPABLE)})"
        )
    factors = [parse_factor(text) for text in args.factor]
    names = [name for name, _ in factors]
    if len(set(names)) != len(names):
        raise SystemExit(f"each factor may appear once; got {names}")

    overrides: Dict[str, Dict[str, object]] = {}
    for combination in itertools.product(*[values for _, values in factors]):
        cell = list(zip(names, combination))
        overrides[matrix_label(cell)] = dict(cell)
    return _variants(args, overrides)


#: Every study, by the name given on the command line.
STUDIES: Dict[str, Callable[[argparse.Namespace], List[Variant]]] = {
    "seeds": study_seeds,
    "reward": study_reward,
    "reward_terms": study_reward_terms,
    "discretizer": study_discretizer,
    "granularity": study_granularity,
    "ablation": study_ablation,
    "params": study_params,
    "sweep": study_sweep,
    "matrix": study_matrix,
}


def build_specs(
    variants: Sequence[Variant], runs_root: Path
) -> List[Dict[str, object]]:
    """Turns labelled configs into job descriptions.

    The directory name carries variant and seed, so ``plot.py`` and
    ``summarize.py`` can recover the grouping from the filesystem alone. No
    index file has to stay in sync with it.
    """
    specs = []
    for label, config in variants:
        name = f"{label}_seed{config.seed}"
        specs.append(
            {
                "label": name,
                "variant": label,
                "run_dir": str(runs_root / name),
                "config": replace(config, run_name=name).to_dict(),
            }
        )
    return specs


def completed_summary(spec: Dict[str, object]) -> Optional[Dict]:
    """The summary of an already finished run, if it matches ``spec`` exactly.

    A run only counts as done when its stored config is identical to the
    requested one, after a JSON round trip. Anything looser would let a
    study silently mix results from two different settings.

    Resuming a run midway is not supported: a checkpoint holds the table but
    not the schedule positions or the RNG state, so continuing from one
    wouldn't reproduce the run it claims to be.
    """
    run_dir = Path(str(spec["run_dir"]))
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    if not (summary_path.exists() and config_path.exists()):
        return None
    stored = json.loads(config_path.read_text(encoding="utf-8"))
    if stored != json.loads(json.dumps(spec["config"])):
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _run_one(spec: Dict[str, object]) -> Dict:
    """Worker: trains one run, or returns the summary of a finished one."""
    done = completed_summary(spec)
    if done is not None:
        return {**done, "skipped": True}
    config = QLearningConfig.from_dict(dict(spec["config"]))  # type: ignore[arg-type]
    summary = train(config, Path(str(spec["run_dir"])), verbose=False)
    return {**summary, "skipped": False}


def run_grid(
    specs: Sequence[Dict[str, object]], runs_root: Path, workers: int
) -> List[Dict]:
    """Runs every spec in a process pool and collects the summaries."""
    runs_root.mkdir(parents=True, exist_ok=True)
    with open(runs_root / "study.json", "w", encoding="utf-8") as handle:
        json.dump(list(specs), handle, indent=2)

    if workers <= 0:
        # leave one core for the rest of the machine, a study shouldn't make
        # the laptop it runs on unusable
        workers = max(1, (multiprocessing.cpu_count() or 2) - 1)

    summaries: List[Dict] = []

    def report(summary: Dict) -> None:
        mark = " (uebersprungen)" if summary.get("skipped") else ""
        print(
            f"[{len(summaries)}/{len(specs)}] {summary['run_name']}{mark}", flush=True
        )

    if workers == 1:
        for spec in specs:
            summaries.append(_run_one(spec))
            report(summaries[-1])
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, spec): spec for spec in specs}
            for future in as_completed(futures):
                summaries.append(future.result())
                report(summaries[-1])

    with open(runs_root / "summaries.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a multi-seed study of the tabular Q-learning agent."
    )
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument("--seeds", type=int, default=5, help="runs per variant")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=0, help="0 means all cores but one"
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--param", default=None, help="field to vary ('sweep')")
    parser.add_argument("--values", nargs="*", default=[], help="values ('sweep')")
    parser.add_argument(
        "--factor",
        action="append",
        default=[],
        metavar="NAME=v1,v2",
        help="one axis of the 'matrix' study; repeat for more axes",
    )
    add_config_arguments(parser)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    variants = STUDIES[args.study](args)
    runs_root = Path(args.runs_root) / f"study_{args.study}"
    specs = build_specs(variants, runs_root)
    print(
        f"Studie '{args.study}': {len(set(s['variant'] for s in specs))} Varianten "
        f"x {args.seeds} Seeds = {len(specs)} Laeufe -> {runs_root}"
    )
    run_grid(specs, runs_root, args.workers)


if __name__ == "__main__":
    main()
