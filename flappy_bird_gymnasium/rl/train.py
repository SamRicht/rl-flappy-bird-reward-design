"""Training loop for the PPO agent, with logging of the learning progress.

Run a single training from the command line, e.g.::

    python -m flappy_bird_gymnasium.rl.train --reward sparse --seed 0 \
        --total-steps 500000

Every run writes into its own directory under ``runs/``:

``config.json``
    All three configurations (PPO, reward, environment), so a run is fully
    reproducible from its output alone.
``episodes.csv``
    One row per finished episode -- the raw material for learning curves.
``progress.csv``
    One row per PPO update, with the optimisation diagnostics.
``model.pt``
    The final network weights (plus periodic checkpoints).
"""

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import torch

from flappy_bird_gymnasium.rl.envs import EnvConfig, make_vector_env
from flappy_bird_gymnasium.rl.ppo import (
    ActorCritic,
    PPOConfig,
    RolloutBuffer,
    ppo_update,
)
from flappy_bird_gymnasium.rl.rewards import PRESETS, RewardConfig

#: Episodes averaged before a checkpoint may be declared the best so far.
BEST_SCORE_WINDOW = 100

EPISODE_FIELDS = ["global_step", "env_id", "return", "length", "score", "flap_rate"]
PROGRESS_FIELDS = [
    "update",
    "global_step",
    "sps",
    "learning_rate",
    "ep_return_mean",
    "ep_score_mean",
    "ep_score_max",
    "ep_length_mean",
    "ep_flap_rate_mean",
    "n_episodes",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "epochs_run",
    "stopped_early",
]


class CsvLogger:
    """Appends rows to a CSV file, writing the header on first use."""

    def __init__(self, path: Path, fields: List[str]) -> None:
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader()

    def write(self, row: Dict[str, object]) -> None:
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()


def set_seed(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def limit_torch_threads(n_threads: int = 1) -> None:
    """Restricts intra-op parallelism.

    The networks here are far too small for multi-threaded BLAS to pay off, and
    when several runs of a study execute in parallel the default (one thread per
    core, per process) oversubscribes the CPU badly.
    """
    torch.set_num_threads(n_threads)


def train(
    ppo_config: PPOConfig,
    reward_config: RewardConfig,
    env_config: EnvConfig,
    run_dir: Path,
    device: torch.device = torch.device("cpu"),
    checkpoint_every: int = 0,
    log_every: int = 10,
    verbose: bool = True,
    torch_threads: int = 1,
) -> Dict[str, object]:
    """Trains a PPO agent and records its learning progress.

    Args:
        ppo_config: Algorithm hyperparameters.
        reward_config: The reward scheme to train under.
        env_config: Environment settings.
        run_dir: Directory to write logs, configs and checkpoints into.
        device: Torch device; CPU is the faster choice for this small MLP.
        checkpoint_every: Save intermediate weights every N updates; ``0``
            disables intermediate checkpoints.
        log_every: Print a progress line every N updates.
        verbose: Whether to print progress at all.
        torch_threads: Intra-op thread limit; keep at 1 when running a study.

    Returns:
        A summary dictionary with the final metrics of the run.
    """
    limit_torch_threads(torch_threads)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "ppo": ppo_config.to_dict(),
                "reward": reward_config.to_dict(),
                "env": env_config.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    set_seed(ppo_config.seed)
    envs = make_vector_env(
        env_config, reward_config, ppo_config.n_envs, seed=ppo_config.seed
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    n_actions = int(envs.single_action_space.n)

    model = ActorCritic(
        obs_dim,
        n_actions,
        hidden_sizes=ppo_config.hidden_sizes,
        shared_backbone=ppo_config.shared_backbone,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.lr, eps=1e-5)
    buffer = RolloutBuffer(ppo_config.n_steps, ppo_config.n_envs, obs_dim, device)
    rng = np.random.default_rng(ppo_config.seed)

    episode_logger = CsvLogger(run_dir / "episodes.csv", EPISODE_FIELDS)
    progress_logger = CsvLogger(run_dir / "progress.csv", PROGRESS_FIELDS)

    next_obs_np, _ = envs.reset(seed=ppo_config.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(ppo_config.n_envs, device=device)

    # Running per-environment accumulators for the episode statistics.
    ep_return = np.zeros(ppo_config.n_envs)
    ep_length = np.zeros(ppo_config.n_envs, dtype=np.int64)

    global_step = 0
    start_time = time.time()
    # Episodes finished since the last progress row.
    window: List[Dict[str, float]] = []
    # Scores of the most recent episodes, used to pick the best checkpoint.
    # A single update's window is far too noisy for that: late in training it
    # can hold a single lucky episode, and the maximum over hundreds of such
    # windows is biased upwards by construction.
    recent_scores: Deque[float] = deque(maxlen=BEST_SCORE_WINDOW)
    best_score_mean = -np.inf
    summary: Dict[str, object] = {}

    for update in range(1, ppo_config.n_updates + 1):
        if ppo_config.anneal_lr:
            # Linear decay to zero: large steps early, fine-tuning late.
            frac = 1.0 - (update - 1.0) / ppo_config.n_updates
            optimizer.param_groups[0]["lr"] = frac * ppo_config.lr
        current_lr = optimizer.param_groups[0]["lr"]

        # ------------------------------------------------------------------
        # 1. Collect a rollout with the current policy.
        # ------------------------------------------------------------------
        for step in range(ppo_config.n_steps):
            global_step += ppo_config.n_envs
            buffer.obs[step] = next_obs
            buffer.dones[step] = next_done

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(next_obs)
            buffer.actions[step] = action
            buffer.log_probs[step] = log_prob
            buffer.values[step] = value

            next_obs_np, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            buffer.rewards[step] = torch.as_tensor(
                reward, dtype=torch.float32, device=device
            )

            ep_return += reward
            ep_length += 1

            # A truncated episode continues in principle; bootstrap its value
            # from the observation the agent would have seen next.
            buffer.truncation_values[step] = 0.0
            if truncated.any():
                final_obs = np.stack(list(infos["final_obs"][truncated]))
                with torch.no_grad():
                    values = model.get_value(
                        torch.as_tensor(final_obs, dtype=torch.float32, device=device)
                    )
                buffer.truncation_values[step][
                    torch.as_tensor(truncated, device=device)
                ] = values

            if done.any():
                scores = infos["final_info"]["score"]
                flaps = infos["final_info"]["flaps"]
                for env_id in np.flatnonzero(done):
                    record = {
                        "global_step": global_step,
                        "env_id": int(env_id),
                        "return": float(ep_return[env_id]),
                        "length": int(ep_length[env_id]),
                        "score": int(scores[env_id]),
                        "flap_rate": float(flaps[env_id]) / max(1, ep_length[env_id]),
                    }
                    episode_logger.write(record)
                    window.append(record)
                    recent_scores.append(float(record["score"]))
                ep_return[done] = 0.0
                ep_length[done] = 0

            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

        # ------------------------------------------------------------------
        # 2. Estimate advantages and 3. optimise.
        # ------------------------------------------------------------------
        with torch.no_grad():
            last_value = model.get_value(next_obs)
        advantages, returns = buffer.compute_advantages(
            last_value, next_done, ppo_config.gamma, ppo_config.gae_lambda
        )

        metrics = ppo_update(
            model,
            optimizer,
            buffer.flatten(),
            advantages.reshape(-1),
            returns.reshape(-1),
            ppo_config,
            rng,
        )

        # ------------------------------------------------------------------
        # 4. Log.
        # ------------------------------------------------------------------
        sps = global_step / max(1e-9, time.time() - start_time)
        if window:
            scores = [r["score"] for r in window]
            row = {
                "ep_return_mean": float(np.mean([r["return"] for r in window])),
                "ep_score_mean": float(np.mean(scores)),
                "ep_score_max": int(np.max(scores)),
                "ep_length_mean": float(np.mean([r["length"] for r in window])),
                "ep_flap_rate_mean": float(np.mean([r["flap_rate"] for r in window])),
                "n_episodes": len(window),
            }
        else:
            row = {
                "ep_return_mean": "",
                "ep_score_mean": "",
                "ep_score_max": "",
                "ep_length_mean": "",
                "ep_flap_rate_mean": "",
                "n_episodes": 0,
            }

        progress_logger.write(
            {
                "update": update,
                "global_step": global_step,
                "sps": round(sps, 1),
                "learning_rate": current_lr,
                **row,
                **metrics,
            }
        )

        # Only judge once the buffer is full, so early updates -- when episodes
        # are short and plentiful -- cannot win on a handful of samples.
        if len(recent_scores) == recent_scores.maxlen:
            smoothed = float(np.mean(recent_scores))
            if smoothed > best_score_mean:
                best_score_mean = smoothed
                torch.save(
                    {"model": model.state_dict(), "ppo": ppo_config.to_dict()},
                    run_dir / "model_best.pt",
                )

        if verbose and (update % log_every == 0 or update == 1):
            score = row["ep_score_mean"]
            score_str = f"{score:6.2f}" if isinstance(score, float) else "   n/a"
            print(
                f"[{reward_config.name}|seed {ppo_config.seed}] "
                f"update {update:4d}/{ppo_config.n_updates} "
                f"step {global_step:8d}  score {score_str}  "
                f"entropy {metrics['entropy']:.3f}  "
                f"kl {metrics['approx_kl']:.4f}  {sps:5.0f} steps/s",
                flush=True,
            )

        if checkpoint_every and update % checkpoint_every == 0:
            torch.save(
                {"model": model.state_dict(), "ppo": ppo_config.to_dict()},
                run_dir / f"model_{update:05d}.pt",
            )

        window = []

    torch.save(
        {"model": model.state_dict(), "ppo": ppo_config.to_dict()},
        run_dir / "model.pt",
    )
    episode_logger.close()
    progress_logger.close()
    envs.close()

    summary = {
        "run_dir": str(run_dir),
        "reward": reward_config.name,
        "seed": ppo_config.seed,
        "global_step": global_step,
        "best_score_mean": float(best_score_mean),
        "wall_time_s": round(time.time() - start_time, 1),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reward",
        default="legacy",
        choices=sorted(PRESETS),
        help="Reward preset to train under.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--n-minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--shared-backbone", action="store_true")
    parser.add_argument("--no-anneal-lr", action="store_true")
    parser.add_argument("--pipe-gap", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=3_000)
    parser.add_argument("--use-lidar", action="store_true")
    parser.add_argument(
        "--normalize-reward",
        action="store_true",
        help="Divide rewards by a running return std, removing the scale "
        "difference between reward schemes.",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def config_from_args(args: argparse.Namespace) -> tuple:
    """Translates parsed CLI arguments into the three configuration objects."""
    reward_config = RewardConfig.preset(args.reward)
    ppo_config = PPOConfig(
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches,
        lr=args.lr,
        anneal_lr=not args.no_anneal_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        hidden_sizes=tuple(args.hidden_sizes),
        shared_backbone=args.shared_backbone,
        seed=args.seed,
    )
    env_config = EnvConfig(
        use_lidar=args.use_lidar,
        pipe_gap=args.pipe_gap,
        max_episode_steps=args.max_episode_steps,
        normalize_reward=args.normalize_reward,
        normalize_gamma=args.gamma,
    )
    # Keep the shaping discount consistent with the agent's, otherwise the
    # policy-invariance guarantee of potential-based shaping does not hold.
    if reward_config.uses_shaping:
        reward_config = replace(reward_config, shaping_gamma=args.gamma)
    return ppo_config, reward_config, env_config


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    ppo_config, reward_config, env_config = config_from_args(args)

    run_dir = args.run_dir or args.runs_root / f"{args.reward}_seed{args.seed}"
    summary = train(ppo_config, reward_config, env_config, run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
