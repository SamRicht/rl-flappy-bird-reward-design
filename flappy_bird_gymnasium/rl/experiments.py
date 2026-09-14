"""Runs a grid of training runs in parallel and collects their results.

Three studies are predefined:

``reward``
    Every reward preset, repeated over several seeds.  The main experiment.
``sweep``
    One PPO hyperparameter varied over a list of values, everything else fixed.
``gap``
    The environment's ``pipe_gap`` varied, to probe how the difficulty of the
    task interacts with the reward scheme.

Because a single run keeps one core busy for a few minutes, runs are executed in
a process pool; each worker is restricted to a single Torch thread so that the
pool does not oversubscribe the CPU.

Examples::

    python -m flappy_bird_gymnasium.rl.experiments reward --seeds 3 \
        --total-steps 3000000 --workers 5
    python -m flappy_bird_gymnasium.rl.experiments sweep --param ent_coef \
        --values 0.0 0.005 0.01 0.03 --seeds 2
"""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from flappy_bird_gymnasium.rl.envs import EnvConfig
from flappy_bird_gymnasium.rl.ppo import PPOConfig
from flappy_bird_gymnasium.rl.rewards import PRESETS, RewardConfig
from flappy_bird_gymnasium.rl.train import train

#: Hyperparameters the ``sweep`` study is allowed to vary, with their type.
SWEEPABLE: Dict[str, type] = {
    "lr": float,
    "gamma": float,
    "gae_lambda": float,
    "clip_eps": float,
    "ent_coef": float,
    "vf_coef": float,
    "n_steps": int,
    "n_epochs": int,
    "n_minibatches": int,
    "n_envs": int,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["reward", "sweep", "gap"])
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
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    builders = {"reward": study_reward, "sweep": study_sweep, "gap": study_gap}
    configs = builders[args.study](args)

    default_root = {
        "reward": Path("runs/reward_study"),
        "sweep": Path(f"runs/sweep_{args.param}"),
        "gap": Path("runs/gap_study"),
    }[args.study]
    runs_root = args.runs_root or default_root

    workers = args.workers or max(1, (multiprocessing.cpu_count() // 2) - 1)
    specs = build_specs(configs, runs_root, args.log_every)
    run_grid(specs, runs_root, workers)


if __name__ == "__main__":
    main()
