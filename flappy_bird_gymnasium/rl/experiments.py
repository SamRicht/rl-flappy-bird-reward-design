"""Runs many training runs in parallel and collects their results.

Four studies are predefined:

``reward``
    Every reward preset, repeated over several seeds.  The main experiment.
``sweep``
    One PPO hyperparameter varied over a list of values, everything else fixed.
``grid``
    The full cross product of several hyperparameters, so that interactions
    become visible.  Needed whenever two settings act on the same underlying
    quantity, or whenever one is only interpretable alongside another -- see the
    warning about batch size below.
``gap``
    The environment's ``pipe_gap`` varied, to probe how the difficulty of the
    task interacts with the reward scheme.

Because a single run keeps one core busy for a few minutes, runs are executed in
a process pool; each worker is restricted to a single Torch thread so that the
pool does not oversubscribe the CPU.

A warning learned the hard way: ``steps = batch * updates``.  Sweeping the batch
size (``n_steps`` or ``n_envs``) while holding ``total_steps`` fixed silently
divides the number of gradient updates by the same factor, and the result then
measures the missing updates rather than the batch.  Vary ``total_steps``
alongside it and read the diagonal of the resulting grid.

Examples::

    python -m flappy_bird_gymnasium.rl.experiments reward --seeds 3
    python -m flappy_bird_gymnasium.rl.experiments grid --seeds 3
        --grid n_steps=256,1024 total_steps=4000000,16000000
"""

import argparse
import itertools
import json
import multiprocessing
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from flappy_bird_gymnasium.rl.envs import EnvConfig
from flappy_bird_gymnasium.rl.ppo import PPOConfig
from flappy_bird_gymnasium.rl.rewards import PRESETS, RewardConfig
from flappy_bird_gymnasium.rl.train import train


def _to_bool(raw: str) -> bool:
    """Parses a flag value; ``argparse``-style truthiness is too permissive here."""
    lowered = raw.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected a boolean, got {raw!r}")


def _hidden_sizes(raw: str) -> tuple:
    """Parses a network shape: ``"256"`` means two layers of 256, ``"64x32"``
    an explicit stack. Two equal layers are the common case, so the short form
    covers it without ``x``-separated repetition."""
    parts = [int(x) for x in raw.lower().split("x") if x]
    if not parts:
        raise ValueError(f"expected a layer width, got {raw!r}")
    return tuple(parts) if len(parts) > 1 else (parts[0], parts[0])


def _cast(name: str, raw: str):
    """Parses one grid value, accepting ``none`` for the optional settings."""
    caster = SWEEPABLE[name]
    if caster is not bool and raw.lower() in ("none", "null"):
        return None
    if caster is tuple:
        return _hidden_sizes(raw)
    return _to_bool(raw) if caster is bool else caster(raw)


def _parse_grid(specs: Sequence[str]) -> "OrderedDict[str, list]":
    """Turns ``["n_steps=256,1024", "target_kl=none,0.02"]`` into an axis map."""
    axes: "OrderedDict[str, list]" = OrderedDict()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"expected name=v1,v2 but got {spec!r}")
        name, raw = spec.split("=", 1)
        if name not in SWEEPABLE:
            raise KeyError(
                f"{name!r} is not sweepable; choose from {sorted(SWEEPABLE)}"
            )
        axes[name] = [_cast(name, v) for v in raw.split(",")]
    return axes


def _variant_name(reward: str, combo: Dict[str, object]) -> str:
    """Builds a run label. Must not contain '_seed', which separates the seed."""
    parts = [reward]
    for key, value in combo.items():
        if value is None:
            shown = "off"
        elif isinstance(value, tuple):
            shown = "x".join(str(v) for v in value)
        else:
            shown = str(value)
        parts.append(f"{key.replace('_', '')}{shown}")
    return "-".join(parts)


#: Hyperparameters the ``sweep`` study is allowed to vary, with their type.
SWEEPABLE: Dict[str, type] = {
    "lr": float,
    "gamma": float,
    "gae_lambda": float,
    "clip_eps": float,
    "ent_coef": float,
    "n_steps": int,
    "n_epochs": int,
    "n_minibatches": int,
    "n_envs": int,
    "target_kl": float,
    # Varying the data budget alongside the batch is what makes a batch-size
    # comparison interpretable: steps = batch * updates, so holding the steps
    # fixed silently divides the number of gradient updates by the same factor.
    "total_steps": int,
    "anneal_lr": bool,
    "norm_adv": bool,
    "clip_vloss": bool,
    "shared_backbone": bool,
    "vf_coef": float,
    "max_grad_norm": float,
    # Widths are given as "256" (two equal layers) or "64x32" (explicit stack).
    "hidden_sizes": tuple,
}


def _run_one(spec: Dict[str, object]) -> Dict[str, object]:
    """Worker entry point: trains one configuration and returns its summary.

    Takes plain dictionaries rather than dataclasses so the payload pickles
    cleanly across process boundaries.
    """
    ppo_config = PPOConfig.from_dict(spec["ppo"])
    reward_config = RewardConfig.from_dict(spec["reward"])
    env_config = EnvConfig.from_dict(spec["env"])
    run_dir = Path(spec["run_dir"])

    summary = train(
        ppo_config,
        reward_config,
        env_config,
        run_dir,
        log_every=spec.get("log_every", 50),
        verbose=spec.get("verbose", True),
        torch_threads=1,
    )
    summary["label"] = spec["label"]
    summary["variant"] = spec["variant"]
    return summary


def build_specs(
    labels_and_configs: Sequence[tuple],
    runs_root: Path,
    log_every: int,
) -> List[Dict[str, object]]:
    """Turns ``(variant, ppo, reward, env)`` tuples into picklable job specs."""
    specs = []
    for variant, ppo_config, reward_config, env_config in labels_and_configs:
        label = f"{variant}_seed{ppo_config.seed}"
        specs.append(
            {
                "label": label,
                "variant": variant,
                "run_dir": str(runs_root / label),
                "ppo": ppo_config.to_dict(),
                "reward": reward_config.to_dict(),
                "env": env_config.to_dict(),
                "log_every": log_every,
            }
        )
    return specs


def run_grid(
    specs: List[Dict[str, object]],
    runs_root: Path,
    workers: int,
) -> List[Dict[str, object]]:
    """Executes the given job specs in a process pool.

    Args:
        specs: Job specifications from :func:`build_specs`.
        runs_root: Directory the study writes into.
        workers: Number of concurrent training processes.

    Returns:
        The summaries of all runs, in completion order.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "study.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")

    print(f"running {len(specs)} runs on {workers} workers -> {runs_root}")
    start = time.time()
    summaries: List[Dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, spec): spec for spec in specs}
        for done, future in enumerate(as_completed(futures), start=1):
            summary = future.result()
            summaries.append(summary)
            elapsed = time.time() - start
            print(
                f"  [{done}/{len(specs)}] {summary['label']}: "
                f"best mean score {summary['best_score_mean']:.2f} "
                f"({elapsed / 60:.1f} min elapsed)",
                flush=True,
            )

    (runs_root / "summaries.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(f"done in {(time.time() - start) / 60:.1f} min")
    return summaries


def base_configs(args: argparse.Namespace) -> tuple:
    """Builds the configurations shared by every run of a study."""
    ppo_config = PPOConfig(
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        hidden_sizes=tuple(args.hidden_sizes),
    )
    env_config = EnvConfig(
        pipe_gap=args.pipe_gap,
        max_episode_steps=args.max_episode_steps,
        normalize_reward=args.normalize_reward,
        normalize_gamma=args.gamma,
    )
    return ppo_config, env_config


def _reward_for(name: str, gamma: float) -> RewardConfig:
    """Returns a preset with its shaping discount tied to the agent's gamma."""
    config = RewardConfig.preset(name)
    if config.uses_shaping:
        config = replace(config, shaping_gamma=gamma)
    return config


def study_reward(args: argparse.Namespace) -> List[tuple]:
    """All (or the selected) reward presets, over several seeds."""
    ppo_config, env_config = base_configs(args)
    names = args.rewards or sorted(PRESETS)
    return [
        (
            name,
            replace(ppo_config, seed=seed),
            _reward_for(name, args.gamma),
            env_config,
        )
        for name in names
        for seed in range(args.seeds)
    ]


def study_sweep(args: argparse.Namespace) -> List[tuple]:
    """One hyperparameter varied over the given values, over several seeds."""
    ppo_config, env_config = base_configs(args)
    caster = SWEEPABLE[args.param]
    reward_config = _reward_for(args.reward, args.gamma)

    specs = []
    for raw in args.values:
        value = caster(raw)
        variant = f"{args.param}{value}"
        for seed in range(args.seeds):
            candidate = replace(ppo_config, seed=seed, **{args.param: value})
            env = env_config
            if args.param == "gamma":
                # Keep the shaping discount and the reward normaliser aligned.
                reward_config = _reward_for(args.reward, value)
                env = replace(env_config, normalize_gamma=value)
            specs.append((variant, candidate, reward_config, env))
    return specs


def study_gap(args: argparse.Namespace) -> List[tuple]:
    """The environment's pipe gap varied, over several seeds."""
    ppo_config, env_config = base_configs(args)
    reward_config = _reward_for(args.reward, args.gamma)
    return [
        (
            f"gap{gap}",
            replace(ppo_config, seed=seed),
            reward_config,
            replace(env_config, pipe_gap=gap),
        )
        for gap in args.gaps
        for seed in range(args.seeds)
    ]


def study_grid(args: argparse.Namespace) -> List[tuple]:
    """Full cross product of several hyperparameters, over reward presets/seeds.

    Unlike ``sweep``, which moves one knob at a time, this covers interactions --
    a larger batch and a KL brake are expected to work on the same underlying
    problem (gradient noise), so varying them independently would not show
    whether they add up.
    """
    ppo_config, env_config = base_configs(args)
    axes = _parse_grid(args.grid)
    rewards = args.rewards or [args.reward]

    specs = []
    for reward_name in rewards:
        reward_config = _reward_for(reward_name, args.gamma)
        for values in itertools.product(*axes.values()):
            combo = dict(zip(axes.keys(), values))
            variant = _variant_name(reward_name, combo)
            for seed in range(args.seeds):
                specs.append(
                    (
                        variant,
                        replace(ppo_config, seed=seed, **combo),
                        reward_config,
                        env_config,
                    )
                )
    return specs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["reward", "sweep", "gap", "grid"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0, help="0 = cores - 1")
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--n-minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--pipe-gap", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=3_000)
    parser.add_argument("--normalize-reward", action="store_true")

    parser.add_argument(
        "--rewards",
        nargs="+",
        default=None,
        choices=sorted(PRESETS),
        help="Subset of presets for the 'reward' study (default: all).",
    )
    parser.add_argument(
        "--reward",
        default="legacy",
        choices=sorted(PRESETS),
        help="Fixed reward preset for the 'sweep' and 'gap' studies.",
    )
    parser.add_argument("--param", choices=sorted(SWEEPABLE), default="ent_coef")
    parser.add_argument("--values", nargs="+", default=["0.0", "0.01", "0.03"])
    parser.add_argument("--gaps", type=int, nargs="+", default=[80, 100, 120])
    parser.add_argument(
        "--grid",
        nargs="+",
        default=["n_steps=256,1024", "target_kl=none,0.02"],
        help="Axes for the 'grid' study as name=v1,v2 (use 'none' to disable an "
        "optional setting).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    builders = {
        "reward": study_reward,
        "sweep": study_sweep,
        "gap": study_gap,
        "grid": study_grid,
    }
    configs = builders[args.study](args)

    default_root = {
        "reward": Path("runs/reward_study"),
        "sweep": Path(f"runs/sweep_{args.param}"),
        "gap": Path("runs/gap_study"),
        "grid": Path("runs/grid_study"),
    }[args.study]
    runs_root = args.runs_root or default_root

    workers = args.workers or max(1, (multiprocessing.cpu_count() // 2) - 1)
    specs = build_specs(configs, runs_root, args.log_every)
    run_grid(specs, runs_root, workers)


if __name__ == "__main__":
    main()
