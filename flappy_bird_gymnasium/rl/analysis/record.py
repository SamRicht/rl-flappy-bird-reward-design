"""Records one greedy episode of a checkpoint as an animated GIF.

Watching the agent is the quickest way to tell *how* it solves the task: the
score says it survives, the recording says whether it glides through the
centre of each gap or scrapes past the edges.

**A recording is an illustration, not a measurement.** It shows one episode
of one seed; the number a report gives is still the one from ``summarize.py``.
That's why a ``.txt`` next to the GIF says which run, which episode seed and
which stretch of the episode it shows.

A strong policy plays for tens of thousands of frames, so ``--seconds`` caps
the clip and ``--skip-to-pipe`` starts it later: the first pipes look the same
for every agent. To replay a measured episode exactly, pass the measurement's
episode seed and frame limit.

GIF because Pillow is the only image library the project already has (it's
installed alongside matplotlib), and a GIF plays in slides and browsers with no
codec question.

Usage::

    python -m flappy_bird_gymnasium.rl.analysis.record runs/q1/best.npz \\
        --algorithm qlearning --out docs/q1.gif --seconds 20 --skip-to-pipe 5
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from flappy_bird_gymnasium.rl.analysis.runs import (
    LOADERS,
    RunLoader,
    open_env,
    resolve_loader,
)
from flappy_bird_gymnasium.rl.rollout import EVAL_SEED

#: The environment renders at 30 fps; the GIF keeps it so motion looks right.
RENDER_FPS = 30

#: A strong policy may never end an episode on its own.
DEFAULT_MAX_STEPS = 2_000_000


def _score_font(size: int):
    """A scalable font for the score, with a fallback chain.

    Pillow's built-in font is a fixed-size bitmap, unreadably small on a
    288-pixel frame.
    """
    from PIL import ImageFont

    candidates: List[Path] = []
    try:
        import matplotlib

        candidates.append(
            Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
        )
    except ImportError:
        pass
    candidates += [
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _overlay_score(image, score: int):
    """Draws the score at the top of the frame.

    Not through the game's own sprites: ``render()`` leaves the score out in
    ``rgb_array`` mode. Drawn after scaling, so it stays sharp at the final
    resolution.
    """
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    text = str(score)
    font = _score_font(max(14, int(image.width * 0.12)))
    left, top, right, _ = draw.textbbox((0, 0), text, font=font)
    x = (image.width - (right - left)) / 2 - left
    y = image.height * 0.06 - top
    # an outline, so the digits read over bright sky and dark pipes alike
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=(20, 20, 20))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return image


def record(
    checkpoint: Path,
    loader: RunLoader,
    out: Path,
    seconds: float = 20.0,
    seed: int = EVAL_SEED,
    every: int = 1,
    scale: float = 1.0,
    skip_to_pipe: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
    with_score: bool = True,
) -> Dict[str, object]:
    """Plays one greedy episode and writes it as a GIF plus a ``.txt``.

    Args:
        checkpoint: The checkpoint file.
        loader: How to read this method's checkpoints.
        out: Destination ``.gif``.
        seconds: Length of the clip in game time.
        seed: Episode seed. ``EVAL_SEED + i`` replays episode ``i`` of a
            ``summarize.py`` measurement.
        every: Keep every n-th frame. Halves the file at ``2`` while the
            clip still covers ``seconds`` of game time, just choppier.
        scale: Resize factor; below 1 shrinks the file.
        skip_to_pipe: Start recording once the score reaches this.
        max_steps: Frame limit of the episode. Set it to a measurement's
            limit to end the clip exactly where that measurement ended.
        with_score: Draw the score into the frames.

    Returns:
        What the clip shows, as also written to the ``.txt``.
    """
    from PIL import Image

    loaded = loader.load(checkpoint)
    env = open_env(loaded, max_episode_steps=max_steps, render_mode="rgb_array")

    # `every` drops frames but covers the same game time, so the number of
    # frames to keep shrinks with it
    wanted = max(1, int(seconds * RENDER_FPS / every))
    frames: List = []
    obs, _ = env.reset(seed=seed)
    score, step, flaps = 0, 0, 0
    start_score: Optional[int] = None
    terminated = truncated = False

    while len(frames) < wanted:
        action = int(loaded.policy(obs))
        obs, _, terminated, truncated, info = env.step(action)
        score = int(info.get("score", score))
        flaps += action
        step += 1
        if score >= skip_to_pipe:
            if start_score is None:
                start_score = score
            if step % every == 0:
                image = Image.fromarray(np.asarray(env.render(), dtype=np.uint8))
                if scale != 1.0:
                    image = image.resize(
                        (int(image.width * scale), int(image.height * scale)),
                        Image.LANCZOS,
                    )
                if with_score:
                    image = _overlay_score(image, score)
                # an adaptive palette keeps the sprites clean at a fraction of
                # the size; the game only uses a handful of hues anyway
                frames.append(image.convert("P", palette=Image.ADAPTIVE, colors=128))
        if terminated or truncated:
            break
    env.close()

    if not frames:
        raise SystemExit(
            f"no frames recorded: the episode ended at score {score}, "
            f"before --skip-to-pipe {skip_to_pipe}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 * every / RENDER_FPS),
        loop=0,
        optimize=True,
    )

    stats: Dict[str, object] = {
        "algorithm": loader.algorithm,
        "checkpoint": str(checkpoint),
        "run": checkpoint.parent.name,
        "reward_preset": loaded.config.reward_preset,
        "episode_seed": seed,
        "frames": len(frames),
        "seconds": round(len(frames) * every / RENDER_FPS, 1),
        "pipes_in_clip": score - (start_score or 0),
        "score_at_end_of_clip": score,
        "episode_ended": bool(terminated or truncated),
        "crashed": bool(terminated),
        "flap_rate": round(flaps / max(step, 1), 3),
        "megabytes": round(out.stat().st_size / 1e6, 2),
    }
    out.with_suffix(".txt").write_text(
        "Recording of a single run: an illustration, not a measurement.\n"
        "The number to report is the one from summarize.py, across seeds.\n\n"
        + "\n".join(f"{k}: {v}" for k, v in stats.items())
        + "\n",
        encoding="utf-8",
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a greedy episode as a GIF.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--algorithm",
        required=True,
        help=f"one of {sorted(LOADERS)}, or 'package.module:RUN_LOADER'",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument(
        "--every", type=int, default=1, help="keep every n-th frame (smaller file)"
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--skip-to-pipe", type=int, default=0, help="start once the score reaches this"
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--no-score", action="store_true", help="no score overlay")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    stats = record(
        args.checkpoint,
        resolve_loader(args.algorithm),
        args.out,
        seconds=args.seconds,
        seed=args.seed,
        every=args.every,
        scale=args.scale,
        skip_to_pipe=args.skip_to_pipe,
        max_steps=args.max_steps,
        with_score=not args.no_score,
    )
    for key, value in stats.items():
        print(f"  {key:22s} {value}")


if __name__ == "__main__":
    main()
