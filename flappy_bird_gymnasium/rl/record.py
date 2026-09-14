"""Records a trained policy playing, as an animated GIF.

Watching the agent is the quickest way to tell *how* it solves the task -- the
score says it survives, the recording says whether it glides through the centre
of each gap or scrapes past the edges.

A strong policy plays for tens of thousands of frames (the best runs here
average over 1500 pipes, roughly half an hour of game time), so a full episode
is useless as a recording.  ``--max-frames`` caps it, and ``--skip`` lets long
stretches be sampled rather than recorded frame by frame.

Usage::

    python -m flappy_bird_gymnasium.rl.record runs/reward_study_v2/risk_averse_seed5 \
        --out docs/figures_v2/risk_averse_seed5.gif --max-frames 600
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from flappy_bird_gymnasium.rl.envs import make_env
from flappy_bird_gymnasium.rl.evaluate import load_run
from flappy_bird_gymnasium.rl.train import limit_torch_threads

#: The environment renders at this rate; the GIF keeps it so motion looks right.
RENDER_FPS = 30


def record(
    run_dir: Path,
    out_path: Path,
    checkpoint: str = "model.pt",
    seed: int = 10_000,
    max_frames: int = 600,
    skip: int = 1,
    scale: float = 1.0,
    deterministic: bool = False,
) -> dict:
    """Plays one episode and writes it as a GIF.

    Args:
        run_dir: A finished run directory.
        out_path: Destination ``.gif``.
        checkpoint: Which checkpoint to load.
        seed: Episode seed, so the same pipe layout can be replayed.
        max_frames: Stop after this many rendered frames.
        skip: Keep every n-th frame; the playback rate is divided to match, so
            the GIF still runs at real time, just choppier.
        scale: Resize factor applied to every frame.
        deterministic: Take the arg-max action instead of sampling.

    Returns:
        Summary of what was recorded.
    """
    limit_torch_threads(1)
    model, _, reward_config, env_config = load_run(run_dir, checkpoint)
    env = make_env(env_config, reward_config, seed=seed, render_mode="rgb_array")()

    obs, _ = env.reset(seed=seed)
    frames: List[Image.Image] = []
    step = 0
    terminated = truncated = False
    info = {"score": 0}

    while not (terminated or truncated) and len(frames) < max_frames:
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if deterministic:
                logits = model.actor_head(model.actor_body(model.backbone(tensor)))
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                sampled, _, _, _ = model.get_action_and_value(tensor)
                action = int(sampled.item())

        if step % skip == 0:
            frame = env.render()
            image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
            if scale != 1.0:
                image = image.resize(
                    (int(image.width * scale), int(image.height * scale)),
                    Image.LANCZOS,
                )
            # An adaptive palette keeps the sprites clean at a fraction of the
            # size of full colour; the game only uses a handful of hues anyway.
            frames.append(image.convert("P", palette=Image.ADAPTIVE, colors=128))

        obs, _, terminated, truncated, info = env.step(action)
        step += 1

    env.close()

    if not frames:
        raise RuntimeError("no frames captured")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 * skip / RENDER_FPS),
        loop=0,
        optimize=True,
    )

    return {
        "path": str(out_path),
        "frames": len(frames),
        "steps_played": step,
        "seconds_of_play": round(step / RENDER_FPS, 1),
        "score_at_cut": int(info["score"]),
        "episode_ended": bool(terminated or truncated),
        "size_mb": round(out_path.stat().st_size / 1048576, 2),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    result = record(
        args.run_dir,
        args.out,
        checkpoint=args.checkpoint,
        seed=args.seed,
        max_frames=args.max_frames,
        skip=args.skip,
        scale=args.scale,
        deterministic=args.deterministic,
    )
    for key, value in result.items():
        print(f"{key:18} {value}")


if __name__ == "__main__":
    main()
