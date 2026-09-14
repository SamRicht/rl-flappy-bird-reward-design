"""Evaluation of a trained PPO policy.

Besides the plain score, the evaluation collects the behavioural statistics that
make the *learned policy* -- not just its performance -- comparable across reward
schemes: how often the agent flaps, and how far from the centre of the gap it
passes the pipes.  Those are the quantities in which the influence of the reward
design becomes visible.

Usage::

    python -m flappy_bird_gymnasium.rl.evaluate runs/sparse_seed0 --episodes 50
    python -m flappy_bird_gymnasium.rl.evaluate runs/sparse_seed0 --render
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from flappy_bird_gymnasium.envs.constants import PIPE_WIDTH, PLAYER_WIDTH
from flappy_bird_gymnasium.rl.envs import EnvConfig, make_env
from flappy_bird_gymnasium.rl.ppo import ActorCritic, PPOConfig
from flappy_bird_gymnasium.rl.rewards import RewardConfig

#: Defaults of the environment's screen; the bird's x is fixed at 20 % of it.
SCREEN_WIDTH = 288
PLAYER_X = int(SCREEN_WIDTH * 0.2)


def load_run(run_dir: Path, checkpoint: str = "model.pt"):
    """Loads the configurations and network weights of a finished run.

    Args:
        run_dir: Directory produced by :func:`flappy_bird_gymnasium.rl.train.train`.
        checkpoint: File name of the checkpoint to load.

    Returns:
        Tuple of ``(model, ppo_config, reward_config, env_config)``.
    """
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ppo_config = PPOConfig.from_dict(config["ppo"])
    reward_config = RewardConfig.from_dict(config["reward"])
    env_config = EnvConfig.from_dict(config["env"])

    obs_dim = 180 if env_config.use_lidar else 12
    model = ActorCritic(
        obs_dim,
        2,
        hidden_sizes=ppo_config.hidden_sizes,
        shared_backbone=ppo_config.shared_backbone,
    )
    state = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    return model, ppo_config, reward_config, env_config


# Normalised x of the bird's right edge and of the pipes' width, needed to tell
# which of the three pipe slots the bird is currently flying towards.
_PLAYER_RIGHT_NORM = (PLAYER_X + PLAYER_WIDTH) / SCREEN_WIDTH
_PIPE_WIDTH_NORM = PIPE_WIDTH / SCREEN_WIDTH


def _gap_offset(obs: np.ndarray, env_config: EnvConfig) -> Optional[float]:
    """Returns the bird's vertical offset from the centre of the gap it targets.

    Only defined for the 12-feature observation, which holds three pipe slots of
    ``(x, top of gap, bottom of gap)`` sorted by ``x``, followed by the bird's
    position, velocity and rotation -- all normalised by the screen size.

    The slots are sorted by distance, *not* by role: while the leftmost pipe has
    not been passed yet it sits in slot 0 and is the bird's target; afterwards
    slot 0 holds the already-passed pipe and the target moves to slot 1.  The
    nearest slot that still lies ahead of the bird is therefore selected
    explicitly.

    Returns:
        The signed offset in screen heights (positive = bird below the centre),
        or `None` when no slot lies ahead.
    """
    if env_config.use_lidar:
        return None

    slots = [(float(obs[i]), float(obs[i + 1]), float(obs[i + 2])) for i in (0, 3, 6)]
    ahead = [s for s in slots if s[0] + _PIPE_WIDTH_NORM >= _PLAYER_RIGHT_NORM]
    if not ahead:
        return None

    _, gap_top, gap_bottom = min(ahead, key=lambda slot: slot[0])
    return float(obs[9] - (gap_top + gap_bottom) / 2)


def evaluate(
    model: ActorCritic,
    env_config: EnvConfig,
    reward_config: RewardConfig,
    episodes: int = 50,
    seed: int = 10_000,
    deterministic: bool = False,
    render_mode: Optional[str] = None,
    audio_on: bool = False,
    max_steps: Optional[int] = None,
) -> Dict[str, object]:
    """Runs the policy for a number of episodes and aggregates its behaviour.

    Args:
        model: Trained actor-critic.
        env_config: Environment settings; reuse the ones the agent trained on.
        reward_config: Reward scheme, needed so that the reported return matches
            the objective the agent was trained on.
        episodes: Number of episodes to play.
        seed: Base seed; episode ``i`` uses ``seed + i``, so evaluations of
            different agents see exactly the same pipe layouts.
        deterministic: Take the arg-max action instead of sampling.  Sampling
            reflects the policy that was actually optimised; the arg-max variant
            usually scores higher and is reported separately.
        render_mode: ``"human"`` to watch the agent play.
        audio_on: Play sounds while rendering.
        max_steps: Overrides the environment's episode step limit.

    Returns:
        Dictionary of aggregated metrics plus the per-episode raw values.
    """
    if max_steps is not None:
        env_config = EnvConfig.from_dict(
            {**env_config.to_dict(), "max_episode_steps": max_steps}
        )
    env = make_env(
        env_config,
        reward_config,
        seed=seed,
        render_mode=render_mode,
        audio_on=audio_on,
    )()

    scores: List[int] = []
    returns: List[float] = []
    lengths: List[int] = []
    flap_rates: List[float] = []
    truncations = 0
    gap_offsets: List[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total_reward, steps = 0.0, 0
        terminated = truncated = False

        while not (terminated or truncated):
            with torch.no_grad():
                tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                if deterministic:
                    features = model.backbone(tensor)
                    logits = model.actor_head(model.actor_body(features))
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    sampled, _, _, _ = model.get_action_and_value(tensor)
                    action = int(sampled.item())

            offset = _gap_offset(np.asarray(obs), env_config)
            if offset is not None:
                gap_offsets.append(abs(offset))

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        scores.append(int(info["score"]))
        returns.append(total_reward)
        lengths.append(steps)
        flap_rates.append(float(info["flaps"]) / max(1, steps))
        truncations += int(truncated)

    env.close()

    return {
        "episodes": episodes,
        "deterministic": deterministic,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_median": float(np.median(scores)),
        "score_min": int(np.min(scores)),
        "score_max": int(np.max(scores)),
        "return_mean": float(np.mean(returns)),
        "length_mean": float(np.mean(lengths)),
        "flap_rate_mean": float(np.mean(flap_rates)),
        # Mean absolute distance from the centre of the gap, in screen heights.
        # A risk-averse policy keeps this small.
        "gap_offset_mean": float(np.mean(gap_offsets)) if gap_offsets else float("nan"),
        # Fraction of episodes that hit the step limit instead of dying -- with a
        # strong policy this approaches 1 and the score becomes limit-bound.
        "truncation_rate": truncations / episodes,
        "scores": scores,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the episode step limit used during training.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    model, _, reward_config, env_config = load_run(args.run_dir, args.checkpoint)
    result = evaluate(
        model,
        env_config,
        reward_config,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=args.deterministic,
        render_mode="human" if args.render else None,
        audio_on=args.audio,
        max_steps=args.max_steps,
    )

    printable = {k: v for k, v in result.items() if k != "scores"}
    print(json.dumps(printable, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
