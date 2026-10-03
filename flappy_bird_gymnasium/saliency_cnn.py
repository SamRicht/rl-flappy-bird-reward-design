"""Shows what a trained CNN agent looks at: Grad-CAM and input-gradient maps.

Usage:
    python -m flappy_bird_gymnasium.saliency_cnn --model runs/v2/shaped_seed0/best.pt
    python -m flappy_bird_gymnasium.saliency_cnn --model ... --target value --frames 8
    python -m flappy_bird_gymnasium.saliency_cnn --model runs/v2/shaped_seed0/best.pt \
        --compare runs/v2/legacy_seed0/best.pt --target value --out cmp.png

The agent plays one greedy episode; at `--frames` evenly spaced steps two
heatmaps are computed for the *decision*, i.e. the advantage of the chosen
action over the other one, `Q(a*) - Q(other)` (or, with `--target value`, for
the state value `Q(a*)`):

* **Grad-CAM** (Selvaraju et al., 2017) on the last convolution layer: coarse
  7x7 map of which regions of the screen drive the decision. Reads as "the
  network is looking here".
* **Input gradient** `|d(Q(a*) - Q(other)) / d pixel|`, summed over the four
  stacked frames: fine-grained, shows the edges (bird, pipe openings) that
  matter. The share of each of the four frames is printed as well, which tells
  how much the network relies on motion (older frames) vs. the current image.

Both maps are drawn over the current game frame (ground cropped, as the network
sees it) and saved as one PNG.

With `--compare` two networks are put side by side. Two policies drive the
game apart - after a few steps they are in different states, and comparing
their maps row by row would compare different situations. So the episode is
played by `--model` only, its observations are recorded, and *both* networks
are evaluated on exactly those observations. The figure then shows Grad-CAM
of each network plus its Q-values, so the scale of the value estimates can be
compared too. Which network drove the episode is printed in the figure title.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.pixel_wrapper import make_pixel_env
from flappy_bird_gymnasium.tests.cnn_dqn import load_cnn

ACTION_NAMES = {0: "noop", 1: "flap"}


def make_target(kind, model):
    """Returns a function mapping a network input to (scalar target, action).

    "decision": advantage of the greedy action over the other one - explains
        *why this action*. Noisy when both Q-values are nearly equal.
    "value": the greedy Q-value itself - explains what makes the state good
        or bad, regardless of the action.
    """

    def target(x):
        q = model(x)
        a = int(q.argmax(dim=1))
        if kind == "decision":
            return q[0, a] - q[0, 1 - a], a
        return q[0, a], a

    return target


def grad_cam(model, obs, target_fn):
    """Grad-CAM on the last conv layer for the target.

    Returns the (7, 7) map normalised to [0, 1] and the chosen action.
    """
    activations, gradients = {}, {}
    last_relu = model.features[5]  # ReLU after conv3 -> (1, 64, 7, 7)

    def fwd_hook(_, __, out):
        activations["a"] = out
        out.register_hook(lambda g: gradients.__setitem__("g", g))

    handle = last_relu.register_forward_hook(fwd_hook)
    try:
        x = torch.as_tensor(obs).unsqueeze(0)
        model.zero_grad()
        target, action = target_fn(x)
        target.backward()
    finally:
        handle.remove()

    weights = gradients["g"].mean(dim=(2, 3), keepdim=True)  # (1, 64, 1, 1)
    cam = F.relu((weights * activations["a"]).sum(dim=1))[0]  # (7, 7)
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy(), action


def input_gradient(model, obs, target_fn):
    """|d target / d input| per pixel, summed over the frame stack.

    Returns the (84, 84) map normalised to [0, 1] and the share of the total
    gradient mass that falls on each of the four frames (oldest first).
    """
    x = torch.as_tensor(obs).unsqueeze(0).float().requires_grad_(True)
    model.zero_grad()
    # Replicate the forward pass with a float input so the gradient reaches x
    # (the model divides by 255 internally; passing uint8 would cut the graph).
    target, _ = target_fn(x)
    target.backward()
    grad = x.grad[0].abs()  # (4, 84, 84)
    per_frame = grad.sum(dim=(1, 2))
    share = (per_frame / per_frame.sum()).numpy()
    sal = grad.sum(dim=0)
    sal = sal / (sal.max() + 1e-8)
    return sal.numpy(), share


def play_and_collect(env, model, steps):
    """Plays greedily and returns (obs, rgb_frame, q_values) per step."""
    obs, _ = env.reset(seed=0)
    ground_y = int(env.unwrapped._ground["y"])
    records = []
    for _ in range(steps):
        with torch.no_grad():
            q = model(torch.as_tensor(obs).unsqueeze(0))[0].numpy()
        frame = env.render()[:ground_y]  # crop the ground, like the wrapper
        records.append((obs.copy(), frame.copy(), q))
        obs, _, terminated, truncated, _ = env.step(int(q.argmax()))
        if terminated or truncated:
            break
    return records


def upsample(heat, shape):
    """Bilinear upsampling of a 2-D map to (height, width)."""
    t = torch.as_tensor(heat, dtype=torch.float32)[None, None]
    return F.interpolate(t, size=shape, mode="bilinear", align_corners=False)[0, 0]


def run_label(model_file):
    """Short name of a checkpoint for figure titles, e.g. `shaped_seed0`."""
    return os.path.basename(os.path.dirname(os.path.abspath(model_file)))


def compare_figure(plt, records, models, labels, target, picks, driver):
    """Grad-CAM of several networks on the same recorded observations."""
    n, cols = len(picks), 1 + len(models)
    fig, axes = plt.subplots(n, cols, figsize=(2.5 * cols, 2.9 * n), squeeze=False)
    for row, t in enumerate(picks):
        obs, frame = records[t][0], records[t][1]
        h, w = frame.shape[:2]
        ax = axes[row, 0]
        ax.imshow(frame)
        ax.set_title(f"t={t}", fontsize=9)
        line = [f"t={t:<4}"]
        for col, (model, label) in enumerate(zip(models, labels), start=1):
            with torch.no_grad():
                q = model(torch.as_tensor(obs).unsqueeze(0))[0].numpy()
            cam, action = grad_cam(model, obs, make_target(target, model))
            ax = axes[row, col]
            ax.imshow(frame)
            ax.imshow(
                upsample(cam, (h, w)).numpy(),
                cmap="inferno",
                alpha=0.55,
                vmin=0,
                vmax=1,
            )
            ax.set_title(
                f"{label}\n{ACTION_NAMES[action]}  Q={q[0]:.2f}/{q[1]:.2f}",
                fontsize=8,
            )
            line.append(f"{label}: {ACTION_NAMES[action]:4} Q={q[0]:6.2f}/{q[1]:6.2f}")
        print("  ".join(line))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Grad-CAM ({target}) on the same observations, episode played by {driver}",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def main():
    args = _get_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Same observation model the checkpoint was trained with (see its
    # config.json); a mismatch shows a policy inputs it has never seen.
    env = make_pixel_env(
        score_limit=args.score_limit,
        stride=args.frame_stride,
        size=tuple(args.obs_size[::-1]),
    )
    model = load_cnn(
        args.model, int(env.action_space.n), input_hw=env.observation_space.shape[1:]
    )

    records = play_and_collect(env, model, args.steps)
    env.close()
    target_fn = make_target(args.target, model)
    n = min(args.frames, len(records))
    picks = np.linspace(0, len(records) - 1, n).astype(int)
    print(f"episode length {len(records)} steps, showing steps {picks.tolist()}")

    if args.compare:
        other = load_cnn(
            args.compare,
            int(env.action_space.n),
            input_hw=env.observation_space.shape[1:],
        )
        labels = [run_label(args.model), run_label(args.compare)]
        fig = compare_figure(
            plt, records, [model, other], labels, args.target, picks, labels[0]
        )
        fig.savefig(args.out, dpi=args.dpi)
        print(f"saved {args.out}")
        return

    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.9 * n), squeeze=False)
    for row, t in enumerate(picks):
        obs, frame, q = records[t]
        cam, action = grad_cam(model, obs, target_fn)
        sal, share = input_gradient(model, obs, target_fn)
        h, w = frame.shape[:2]
        cam_up = upsample(cam, (h, w)).numpy()
        sal_up = upsample(sal, (h, w)).numpy()
        adv = q[action] - q[1 - action]
        title = f"t={t}  {ACTION_NAMES[action]}  Q={q[0]:.2f}/{q[1]:.2f}"
        print(
            f"{title}  frame share (old->new): " + " ".join(f"{s:.2f}" for s in share)
        )

        what = f"adv {adv:+.2f}" if args.target == "decision" else f"Q {q[action]:.2f}"
        panels = [
            (None, title),
            (cam_up, f"Grad-CAM  ({what})"),
            (sal_up, "input gradient"),
        ]
        for col, (heat, label) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(frame)
            if heat is not None:
                ax.imshow(heat, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
            ax.set_title(label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved {args.out}")


def _get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True, help="checkpoint (.pt)")
    p.add_argument(
        "--compare",
        type=str,
        default=None,
        help="second checkpoint, evaluated on the observations of --model's "
        "episode (same observation model required)",
    )
    p.add_argument("--frames", type=int, default=6, help="rows in the figure")
    p.add_argument(
        "--target",
        choices=("decision", "value"),
        default="decision",
        help="explain the action choice Q(a*)-Q(other), or the value Q(a*)",
    )
    p.add_argument("--steps", type=int, default=400, help="max steps to play")
    p.add_argument("--score-limit", type=int, default=50)
    p.add_argument(
        "--frame-stride", type=int, default=1, help="as in the run's config.json"
    )
    p.add_argument(
        "--obs-size",
        type=int,
        nargs=2,
        default=[84, 84],
        metavar=("H", "W"),
        help="as in the run's config.json",
    )
    p.add_argument("--out", type=str, default=None, help="default: next to model")
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args()
    if args.out is None:
        name = f"saliency_{args.target}.png"
        if args.compare:
            name = f"saliency_vs_{run_label(args.compare)}_{args.target}.png"
        args.out = os.path.join(os.path.dirname(args.model), name)
    return args


if __name__ == "__main__":
    main()
