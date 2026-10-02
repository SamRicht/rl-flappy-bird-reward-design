"""Trains the convolutional Dueling-DQN on pixel observations.

Usage:
    python -m flappy_bird_gymnasium.train_cnn --steps 500000 --reward shaped

Algorithm: Double DQN (van Hasselt et al., 2016) with a dueling head, uniform
experience replay (one frame per step, see `FrameReplayBuffer`), optional
n-step returns (`--n-step`), hard target-network updates, linearly decaying
epsilon-greedy exploration and a linearly decaying learning rate.

The run follows the layout of the shared evaluation chain (`rl/analysis`), so
`summarize`, `significance`, `plot` and `record` work on it with
`--algorithm cnn` (see `evaluate_cnn`):

    runs/<study>/<reward>_seed<n>/
        config.json    `CnnConfig` plus all arguments and the reward scheme
        train.csv      one row per training episode (score and return apart)
        eval.csv       periodic greedy measurement, `rl.runlog.EVAL_LOG_FIELDS`
        best.pt        checkpoint of the best greedy measurement
        final.pt       checkpoint at the end
        summary.json   written last; marks the run as finished

The environment is the shared one (`rl.rollout.make_env`): same reward
scheme and training frame limit as every other method, seen through the pixel
wrappers. Reaching the frame limit is a truncation and is bootstrapped.

Every `--eval-every` steps the current policy plays greedily (no exploration)
under the training frame limit; see `quick_eval` for why. Numbers for a
report come from `rl.analysis.summarize`, which measures against the much
higher `--eval-max-episode-steps`.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.utils import MODEL_PATH
from flappy_bird_gymnasium.evaluate_cnn import CnnConfig, make_cnn_env
from flappy_bird_gymnasium.rl.rewards import PRESETS
from flappy_bird_gymnasium.rl.rollout import (
    EVAL_SEED,
    reward_config_for,
    rollout,
    set_global_seeds,
    summarize_rollout,
)
from flappy_bird_gymnasium.rl.runlog import (
    EVAL_LOG_FIELDS,
    CsvLogger,
    steps_to_thresholds,
)
from flappy_bird_gymnasium.tests.cnn_dqn import DuelingCNN

DEFAULT_MODEL_FILE = os.path.join(MODEL_PATH, "cnn_dqn.pt")

TRAIN_LOG_FIELDS = [
    "step",
    "episode",
    "return",
    "score",
    "length",
    "flap_rate",
    "epsilon",
    "loss",
]


class FrameReplayBuffer:
    """Uniform replay buffer that stores one frame per time step.

    Instead of the whole 4-frame stack, only the newest frame of every
    observation is kept (~7 KB per transition instead of ~35 KB); the stacks
    are re-assembled when sampling. Positions are absolute time steps `t`
    mapped onto a ring of size `capacity`, so a sample can never straddle the
    write pointer: the valid range is expressed in absolute time and only
    then mapped onto ring indices.

    Episode starts are padded exactly like `FrameStackObservation` does
    (`padding_type="reset"`): the first frame of an episode is repeated, so
    what the network trains on matches what it sees when acting. With a
    temporal `stride` k the stack holds the frames `t, t-k, ..., t-(stack-1)k`,
    mirroring `pixel_wrapper.StridedFrameStack` - the two must agree, or the
    network trains on different stacks than it acts on without any error
    (`test_frame_replay_buffer_reconstructs_env_stacks` guards this).

    n-step returns are computed at sampling time from the stored rewards:
    `ret = sum_k gamma^k r_{t+k}` over the window `k < n_step`, cut off at the
    first episode end. `horizon` is the number of steps actually summed
    (n_step, or less at an episode end), and the bootstrap state is the stack
    at `t + horizon`. For `n_step=1` this is ordinary one-step Double DQN.

    Only a termination (a crash) stops the bootstrap. A truncation - the
    frame limit - does not: the bird was still alive, and treating it as a
    death would teach the network that flying well for long ends badly. The
    bootstrap state of a truncated episode is its final observation, which no
    later step stores (the next frame already belongs to the new episode), so
    `add` appends it as an extra slot that is never sampled as a transition
    start (`valid` is False). One slot per truncated episode, negligible.
    """

    def __init__(self, capacity, obs_shape, device, gamma=0.99, n_step=1, stride=1):
        stack, h, w = obs_shape
        if n_step < 1 or stride < 1:
            raise ValueError(f"n_step and stride must be >= 1, got {n_step}, {stride}")
        self.capacity = capacity
        self.stack = stack
        self.stride = stride
        self.device = device
        self.gamma = gamma
        self.n_step = n_step
        self.frames = np.zeros((capacity, h, w), dtype=np.uint8)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        # done: terminated, no bootstrap. end: episode over, terminated or
        # truncated - cuts the n-step window.
        self.done = np.zeros(capacity, dtype=np.float32)
        self.end = np.zeros(capacity, dtype=bool)
        # False only for the final frame of a truncated episode.
        self.valid = np.zeros(capacity, dtype=bool)
        # Index of the frame within its episode (0 = reset observation).
        self.pos = np.zeros(capacity, dtype=np.int64)
        self.t = 0  # number of frames added so far (absolute time)
        self._prev_end = True

    def __len__(self):
        return min(self.t, self.capacity)

    def add(self, obs, action, reward, terminated, truncated=False, next_obs=None):
        """Stores the transition from `obs`.

        A truncated (and not terminated) step needs `next_obs`, the final
        observation of the episode, as its bootstrap state.
        """
        self._put(obs[-1], action, reward, terminated, terminated or truncated, True)
        if truncated and not terminated:
            if next_obs is None:
                raise ValueError("a truncated step needs next_obs to bootstrap from")
            self._put(next_obs[-1], 0, 0.0, False, True, False)

    def _put(self, frame, action, reward, done, end, valid):
        i = self.t % self.capacity
        self.frames[i] = frame
        self.action[i] = action
        self.reward[i] = reward
        self.done[i] = float(done)
        self.end[i] = end
        self.valid[i] = valid
        # The final frame of a truncated episode still belongs to it.
        new_episode = self._prev_end and valid
        self.pos[i] = 0 if new_episode else self.pos[(i - 1) % self.capacity] + 1
        self._prev_end = bool(end)
        self.t += 1

    def _valid_range(self):
        """Absolute times `[lo, hi)` that can be sampled.

        Needs `(stack - 1) * stride` older frames for the stack (they may
        belong to the previous episode, the padding takes care of that) and
        `n_step` newer frames for the reward window and the bootstrap state.
        """
        oldest = max(0, self.t - self.capacity)
        return oldest + (self.stack - 1) * self.stride, self.t - self.n_step

    def _stacks(self, times):
        """Re-assembles the stacked observation for each absolute time."""
        times = np.asarray(times)
        offsets = np.arange(self.stack - 1, -1, -1) * self.stride  # oldest first
        src = times[:, None] - offsets[None, :]
        pos = self.pos[times % self.capacity]
        # Before the episode start use the reset frame instead.
        src = np.where(
            offsets[None, :] > pos[:, None], times[:, None] - pos[:, None], src
        )
        return self.frames[src % self.capacity]

    def _build(self, times):
        """Builds the batch for absolute times as numpy arrays.

        Returns `(obs, action, ret, next_obs, done, horizon)`.
        """
        times = np.asarray(times)
        window = times[:, None] + np.arange(self.n_step)[None, :]
        end_w = self.end[window % self.capacity].astype(np.float32)  # (batch, n)
        reward_w = self.reward[window % self.capacity]
        # alive[:, k] = 1 while no step before k has ended the episode.
        alive = np.ones_like(end_w)
        alive[:, 1:] = np.cumprod(1.0 - end_w[:, :-1], axis=1)
        discounts = self.gamma ** np.arange(self.n_step, dtype=np.float32)
        ret = (alive * reward_w * discounts[None, :]).sum(axis=1).astype(np.float32)
        horizon = alive.sum(axis=1).astype(np.int64)
        done = self.done[(times + horizon - 1) % self.capacity]
        return (
            self._stacks(times),
            self.action[times % self.capacity],
            ret,
            self._stacks(times + horizon),
            done,
            horizon,
        )

    def sample(self, batch_size):
        lo, hi = self._valid_range()
        times = np.random.randint(lo, hi, size=batch_size)
        # Redraw the final frames of truncated episodes; without truncations
        # this draws nothing extra, so older runs keep their random stream.
        bad = ~self.valid[times % self.capacity]
        while bad.any():
            times[bad] = np.random.randint(lo, hi, size=int(bad.sum()))
            bad = ~self.valid[times % self.capacity]
        to = lambda x: torch.as_tensor(x, device=self.device)  # noqa: E731
        return tuple(to(x) for x in self._build(times))


def linear_epsilon(step, start, end, decay_steps):
    frac = min(1.0, step / decay_steps)
    return start + frac * (end - start)


def select_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def quick_eval(actor, config, episodes, seed=EVAL_SEED):
    """Greedy measurement (epsilon = 0) for `eval.csv` and the pick of `best.pt`.

    The training log is recorded with exploration noise, which in Flappy Bird
    costs a lot (one random flap in front of a pipe is fatal), so this is the
    curve to compare runs on. Episode `i` is reset with `seed + i`: every
    measurement, across runs and reward schemes, sees the same pipes.

    Plays under the *training* frame limit, like the DQN work: a good policy
    would otherwise make this cost more than the training itself. The score
    then saturates once the policy outlives the limit, which
    `truncation_rate` reports. Numbers for a report come from
    `rl.analysis.summarize`, which measures against the higher limit.
    """
    env = make_cnn_env(config)
    measured = rollout(env, actor.get_action, episodes, seed)
    env.close()
    return summarize_rollout(measured)


def train_step(model, target, optimizer, batch, gamma):
    obs, action, ret, next_obs, done, horizon = batch

    q = model(obs).gather(1, action[:, None]).squeeze(1)
    with torch.no_grad():
        # Double DQN: online net picks the action, target net evaluates it.
        next_action = model(next_obs).argmax(dim=1, keepdim=True)
        next_q = target(next_obs).gather(1, next_action).squeeze(1)
        # `ret` already holds the discounted n-step reward sum, so the
        # bootstrap is discounted by gamma^horizon (gamma for one-step).
        y = ret + gamma**horizon * (1.0 - done) * next_q

    loss = F.smooth_l1_loss(q, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def config_from_args(args):
    return CnnConfig(
        reward_preset=args.reward,
        gamma=args.gamma,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        eval_max_episode_steps=args.eval_max_episode_steps,
        frame_stride=args.frame_stride,
        obs_size=tuple(args.obs_size),
    )


def train(args):
    set_global_seeds(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = config_from_args(args)
    # Preset, overrides and the shaping discount tied to `--gamma`, exactly
    # as for every other method.
    reward_config = reward_config_for(config)
    print(f"device: {device}, output: {out}, reward: {reward_config}")

    # Training env and `quick_eval` are built from the same config, so the
    # measurement sees the frames the network was trained on.
    env = make_cnn_env(config)
    n_actions = int(env.action_space.n)
    net_kwargs = dict(
        layer_norm=args.layer_norm,
        orthogonal_init=args.orthogonal_init,
        fc_width=args.fc_width,
        input_hw=env.observation_space.shape[1:],
    )
    model = DuelingCNN(n_actions, **net_kwargs).to(device)
    target = DuelingCNN(n_actions, **net_kwargs).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.adam_eps)

    # Acting on a single observation is dominated by GPU launch/sync overhead
    # (~6 ms on MPS vs ~0.2 ms on CPU), so actions are chosen with a CPU copy
    # of the network that is refreshed every `actor_sync` steps. Acting with a
    # slightly stale policy is harmless for an off-policy method like DQN.
    actor = DuelingCNN(n_actions, **net_kwargs)
    actor.load_state_dict(model.state_dict())
    actor.eval()
    buffer = FrameReplayBuffer(
        args.buffer_size,
        env.observation_space.shape,
        device,
        gamma=args.gamma,
        n_step=args.n_step,
        stride=args.frame_stride,
    )

    # Record the exact reward scheme and hyperparameters next to the results.
    # `config` is what the evaluation chain reads back (`CnnConfig.from_run`).
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config.to_dict(),
                "args": vars(args),
                "reward": reward_config.to_dict(),
            },
            f,
            indent=2,
        )

    train_log = CsvLogger(out / "train.csv", TRAIN_LOG_FIELDS)
    eval_log = CsvLogger(out / "eval.csv", EVAL_LOG_FIELDS)
    best_path = out / "best.pt"
    best_score, best_truncation = -1.0, 0.0

    obs, _ = env.reset(seed=args.seed)
    episode, ep_return, ep_len, ep_losses = 0, 0.0, 0, []
    recent_scores, episode_steps, episode_scores = [], [], []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        epsilon = linear_epsilon(
            step, args.eps_start, args.eps_end, args.eps_decay_steps
        )
        # Linear learning-rate decay over the whole run: the Q-values drift
        # once the policy is decent, a smaller step size late in training
        # keeps them from wandering off.
        for group in optimizer.param_groups:
            group["lr"] = linear_epsilon(step, args.lr, args.lr_end, args.steps)
        if np.random.rand() < epsilon:
            if args.explore_flap_prob is None:
                action = env.action_space.sample()
            else:
                # Uniform sampling flaps 50% of the time; trained agents flap
                # ~6%. Random flaps mostly send the bird into the ceiling, so
                # explore with a flap rate close to the one of a real policy.
                action = 1 if np.random.rand() < args.explore_flap_prob else 0
        else:
            action = actor.get_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        # A truncation is bootstrapped from `next_obs`, see the buffer.
        buffer.add(obs, action, reward, terminated, truncated, next_obs)
        obs = next_obs
        ep_return += reward
        ep_len += 1

        if step > args.learning_starts and step % args.train_every == 0:
            batch = buffer.sample(args.batch_size)
            ep_losses.append(train_step(model, target, optimizer, batch, args.gamma))

        if step % args.target_update == 0:
            target.load_state_dict(model.state_dict())

        if step % args.actor_sync == 0:
            actor.load_state_dict(model.state_dict())

        if terminated or truncated:
            mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            train_log.log(
                {
                    "step": step,
                    "episode": episode,
                    "return": ep_return,
                    "score": info["score"],
                    "length": ep_len,
                    "flap_rate": round(info["flaps"] / max(ep_len, 1), 4),
                    "epsilon": epsilon,
                    "loss": mean_loss,
                }
            )
            recent_scores.append(info["score"])
            episode_steps.append(step)
            episode_scores.append(info["score"])
            if episode % 10 == 0:
                sps = step / (time.time() - t0)
                print(
                    f"step {step:>8} | ep {episode:>5} | "
                    f"score(avg10) {np.mean(recent_scores[-10:]):5.2f} | "
                    f"return {ep_return:7.2f} | eps {epsilon:.3f} | "
                    f"loss {mean_loss:.4f} | {sps:.0f} steps/s"
                )
            episode += 1
            ep_return, ep_len, ep_losses = 0.0, 0, []
            obs, _ = env.reset()

        if step % args.save_every == 0:
            torch.save(model.state_dict(), out / f"ckpt_{step}.pt")

        if step % args.eval_every == 0 and step > args.learning_starts:
            actor.load_state_dict(model.state_dict())
            stats = quick_eval(actor, config, args.eval_episodes, args.eval_seed_base)
            eval_log.log({"step": step, **stats})
            mean = stats["mean_score"]
            marker = ""
            if mean > best_score:
                best_score, best_truncation = mean, stats["truncation_rate"]
                torch.save(model.state_dict(), best_path)
                marker = "  <- new best"
            print(
                f"[eval] step {step:>8} | greedy score mean {mean:6.2f} "
                f"median {stats['median_score']:4.0f} "
                f"min {stats['min_score']:3d} max {stats['max_score']:3d} | "
                f"limit {stats['truncation_rate']:.2f} "
                f"flap {stats['mean_flap_rate']:.3f}{marker}"
            )

    final = out / "final.pt"
    torch.save(model.state_dict(), final)
    print(f"saved {final}")
    if not best_path.exists():
        shutil.copy(final, best_path)
    print(f"best greedy score {best_score:.2f} -> {best_path}")
    if best_truncation >= 1.0:
        print(
            "  (censored by the training frame limit - the real number comes "
            "from rl.analysis.summarize)"
        )
    if not args.no_export:
        os.makedirs(MODEL_PATH, exist_ok=True)
        shutil.copy(best_path, DEFAULT_MODEL_FILE)
        print(f"exported best model to {DEFAULT_MODEL_FILE}")

    train_log.close()
    eval_log.close()
    env.close()

    # Written last: its presence marks the run as finished for the chain.
    summary = {
        "run_dir": str(out),
        "seed": args.seed,
        "reward_preset": args.reward,
        "total_steps": args.steps,
        "episodes": episode,
        "minutes": round((time.time() - t0) / 60, 2),
        "best_eval_score": float(best_score),
        "best_eval_truncation_rate": float(best_truncation),
        **steps_to_thresholds(episode_steps, episode_scores),
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reward",
        type=str,
        default="legacy",
        choices=sorted(PRESETS),
        help="reward preset from flappy_bird_gymnasium/rl/rewards.py",
    )
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument(
        "--buffer-size",
        type=int,
        default=50_000,
        help="transitions in the replay buffer, ~7 KB each (250000 = 1.8 GB)",
    )
    # batch 64 every 8 steps sees as many samples per env step as the classic
    # 32/4, but halves the number of GPU calls (~30% faster on MPS).
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-end", type=float, default=2.5e-5)
    p.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="Adam epsilon (torch default 1e-8). 1.5e-4 is the Atari-DQN value: "
        "it caps the effective step size when gradients get small, the "
        "mechanism behind the post-peak collapses seen in eval.csv",
    )
    # Network ablations, see DuelingCNN for the reasoning behind each.
    p.add_argument(
        "--layer-norm",
        action="store_true",
        help="LayerNorm before the last ReLU of the trunk (damps value divergence)",
    )
    p.add_argument(
        "--orthogonal-init",
        action="store_true",
        help="orthogonal init (gain sqrt(2)) of the trunk, gain 1 for the heads",
    )
    p.add_argument(
        "--fc-width",
        type=int,
        default=512,
        help="width of the linear layer after the conv stack (95%% of the "
        "parameters live there; 256 halves them)",
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument(
        "--n-step",
        type=int,
        default=1,
        help="horizon of the n-step return (1 = plain one-step DQN, try 3)",
    )
    # Observation ablations, see pixel_wrapper.make_pixel_env. Both change
    # what the network sees, so checkpoints are tied to these values.
    p.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="steps between the 4 stacked frames (1 = consecutive); 3 widens "
        "the window to 9 steps so the bird's acceleration becomes visible",
    )
    p.add_argument(
        "--obs-size",
        type=int,
        nargs=2,
        default=[84, 84],
        metavar=("H", "W"),
        help="frame size in pixels; 128 84 keeps more of the vertical axis, "
        "where all the information is",
    )
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=CnnConfig.max_episode_steps,
        help="frame limit per training episode, shared with the other methods; "
        "reaching it is bootstrapped, not treated as a crash",
    )
    p.add_argument(
        "--eval-max-episode-steps",
        type=int,
        default=CnnConfig.eval_max_episode_steps,
        help="frame limit rl.analysis.summarize measures this run with",
    )
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--train-every", type=int, default=8)
    p.add_argument("--target-update", type=int, default=1_000)
    p.add_argument("--actor-sync", type=int, default=250)
    # Random exploration is nearly useless in Flappy Bird (random flaps kill
    # the bird at the ceiling), so start low and end very low.
    p.add_argument("--eps-start", type=float, default=0.1)
    p.add_argument("--eps-end", type=float, default=0.001)
    p.add_argument("--eps-decay-steps", type=int, default=50_000)
    p.add_argument(
        "--explore-flap-prob",
        type=float,
        default=None,
        help="flap probability of a random action (default: uniform, i.e. 0.5; "
        "~0.08 matches the flap rate of trained agents - combine with a higher "
        "--eps-start and longer --eps-decay-steps, e.g. 0.3 and 200000)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="output directory (default: runs/cnn/<reward>_seed<seed>)",
    )
    p.add_argument("--save-every", type=int, default=50_000)
    # 20 episodes every 100k steps cost the same as 5 every 50k but pick
    # `best.pt` with far less selection noise.
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument(
        "--eval-seed-base",
        type=int,
        default=EVAL_SEED,
        help="reset seed of eval episode i is this + i, so all runs are measured "
        "on the same pipe sequences (default: the shared EVAL_SEED)",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--no-export",
        action="store_true",
        help=f"do not copy the final model to {DEFAULT_MODEL_FILE}",
    )
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join("runs", "cnn", f"{args.reward}_seed{args.seed}")
    return args


if __name__ == "__main__":
    train(_get_args())
