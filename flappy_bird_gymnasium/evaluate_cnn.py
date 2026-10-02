"""The CNN agent's side of the shared evaluation chain (`rl/analysis`).

Three things the chain needs from a method, and one it doesn't:

* :class:`CnnConfig`, the run configuration with the attribute names that
  `rl.rollout.make_env` and `rl.analysis.summarize` read;
* :func:`make_cnn_env`, the environment the agent plays in: the shared one
  (reward scheme, frame limit) seen through the pixel wrappers;
* :data:`RUN_LOADER`, registered as ``cnn`` in `rl.analysis.runs.LOADERS`;
* ``backfill``, which brings runs trained before this module existed
  (``log.csv``, old ``eval.csv`` columns, no ``summary.json``) into the run
  layout, so they can be measured next to the new ones.

Usage::

    python -m flappy_bird_gymnasium.rl.analysis.summarize runs/cnn --algorithm cnn
    python -m flappy_bird_gymnasium.rl.analysis.record runs/cnn/shaped_seed0/best.pt \\
        --algorithm cnn --out docs/cnn_shaped.gif
    python -m flappy_bird_gymnasium.evaluate_cnn backfill runs/v2
"""

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass, field, fields
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium

from flappy_bird_gymnasium.envs.pixel_wrapper import FRAME_SIZE, pixel_wrapper
from flappy_bird_gymnasium.rl.analysis.runs import LoadedRun, RunLoader
from flappy_bird_gymnasium.rl.rollout import make_env
from flappy_bird_gymnasium.rl.runlog import steps_to_thresholds


@dataclass
class CnnConfig:
    """What a CNN run's environment and measurement depend on.

    The first block mirrors the DQN work's config, so both methods train and
    get measured under the same environment definition. The second block is
    how the CNN sees that environment; a checkpoint is only valid for the
    values it was trained with.
    """

    # --- environment, shared with the other methods ---
    use_lidar: bool = False
    normalize_obs: bool = True
    pipe_gap: int = 100
    reward_preset: str = "legacy"
    reward_overrides: Dict[str, float] = field(default_factory=dict)
    gamma: float = 0.99
    seed: int = 0
    # Frame limit while training; reaching it is bootstrapped, not a death.
    max_episode_steps: int = 3_000
    # Frame limit while measuring, far higher so good policies aren't censored.
    eval_max_episode_steps: int = 20_000

    # --- observation ---
    frame_stride: int = 1
    obs_size: Tuple[int, int] = (FRAME_SIZE[1], FRAME_SIZE[0])  # (height, width)
    crop_ground: bool = True

    @classmethod
    def from_run(cls, run_dir: Path) -> "CnnConfig":
        """Reads ``config.json`` of a run, old layout included.

        Runs written by `train_cnn` since the shared evaluation carry a
        ``config`` block. Older ones only have the argparse ``args``; their
        observation settings were the defaults unless the flag existed and
        was set.
        """
        data = json.loads((Path(run_dir) / "config.json").read_text("utf-8"))
        if "config" in data:
            raw = data["config"]
        else:
            args = data["args"]
            raw = {
                "reward_preset": args["reward"],
                "gamma": args.get("gamma", 0.99),
                "seed": args["seed"],
                "frame_stride": args.get("frame_stride", 1),
                "obs_size": args.get("obs_size", list(cls.obs_size)),
            }
        known = {f.name for f in fields(cls)}
        config = cls(**{k: v for k, v in raw.items() if k in known})
        config.obs_size = tuple(config.obs_size)
        return config

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def make_cnn_env(
    config: CnnConfig,
    max_episode_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> gymnasium.Env:
    """The shared environment of `rl.rollout.make_env`, as the CNN sees it.

    ``render_mode`` is accepted for the `LoadedRun.make_env` signature, but the
    env is always ``rgb_array``: the pixel wrapper draws every observation
    from it. ``env.render()`` then returns the frame, which is all
    `rl.analysis.record` asks of it.
    """
    del render_mode
    height, width = config.obs_size
    return make_env(
        config,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        wrapper=pixel_wrapper(
            stride=config.frame_stride,
            crop_ground=config.crop_ground,
            size=(width, height),
        ),
        # the bird is nearly invisible in grayscale on the day sky, see
        # `make_pixel_env`
        background="night",
    )


def load(checkpoint: Path) -> LoadedRun:
    """Turns a ``.pt`` checkpoint of a CNN run into a greedy policy."""
    import torch

    from flappy_bird_gymnasium.tests.cnn_dqn import load_cnn

    # `summarize` measures several runs in parallel processes; one thread each
    # keeps them from fighting over the cores.
    torch.set_num_threads(1)
    config = CnnConfig.from_run(Path(checkpoint).parent)
    model = load_cnn(str(checkpoint), input_hw=config.obs_size)
    return LoadedRun(
        config=config,
        policy=model.get_action,
        columns={
            "frame_stride": config.frame_stride,
            "obs_size": "x".join(map(str, config.obs_size)),
        },
        make_env=partial(make_cnn_env, config),
    )


RUN_LOADER = RunLoader(
    algorithm="cnn",
    load=load,
    checkpoints=("best.pt", "final.pt"),
)


# --- backfill of runs from before the shared evaluation ---------------------

#: Old ``eval.csv`` column -> the shared name (`rl.runlog.EVAL_LOG_FIELDS`).
_OLD_EVAL_COLUMNS = {
    "mean": "mean_score",
    "median": "median_score",
    "min": "min_score",
    "max": "max_score",
}


def backfill_run(run_dir: Path) -> Optional[str]:
    """Brings one old run into the shared layout. Returns what was done.

    Only adds files, never loses any: ``train.csv`` is written next to
    ``log.csv``, and an old ``eval.csv`` is kept as ``eval_old.csv`` before
    its columns are renamed. ``summary.json`` is written last, as in a new
    run, and says that it was backfilled: these runs trained without a frame
    limit, so their training curves aren't strictly comparable to new ones.
    Their checkpoints are, which is what ``summarize`` measures.
    """
    log = run_dir / "log.csv"
    if (run_dir / "summary.json").exists() or not log.exists():
        return None
    if not (run_dir / "final.pt").exists():
        return None  # unfinished, leave it to `unfinished_runs` to report

    with log.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    steps: List[int] = []
    scores: List[float] = []
    with (run_dir / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "episode",
                "return",
                "score",
                "length",
                "flap_rate",
                "epsilon",
                "loss",
            ],
        )
        writer.writeheader()
        for row in rows:
            length = int(row["length"])
            writer.writerow(
                {
                    "step": row["step"],
                    "episode": row["episode"],
                    "return": row["return"],
                    "score": row["score"],
                    "length": length,
                    "flap_rate": round(int(row["flaps"]) / max(length, 1), 4),
                    "epsilon": row["epsilon"],
                    "loss": row["loss"],
                }
            )
            steps.append(int(row["step"]))
            scores.append(float(row["score"]))

    best_eval = None
    eval_path = run_dir / "eval.csv"
    if eval_path.exists():
        with eval_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            eval_rows = list(reader)
        if "mean" in header:
            shutil.copy(eval_path, run_dir / "eval_old.csv")
            renamed = [_OLD_EVAL_COLUMNS.get(c, c) for c in header]
            with eval_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(renamed)
                for row in eval_rows:
                    writer.writerow([row[c] for c in header])
            if eval_rows:
                best_eval = max(float(r["mean"]) for r in eval_rows)

    config = CnnConfig.from_run(run_dir)
    summary = {
        "run_dir": str(run_dir),
        "seed": config.seed,
        "reward_preset": config.reward_preset,
        "total_steps": steps[-1] if steps else 0,
        "episodes": len(rows),
        "best_eval_score": best_eval,
        "backfilled": True,
        **steps_to_thresholds(steps, scores),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    return f"train.csv, summary.json{', eval.csv' if best_eval is not None else ''}"


def backfill(study_root: Path) -> None:
    for run_dir in sorted(p for p in study_root.iterdir() if p.is_dir()):
        done = backfill_run(run_dir)
        print(f"  {run_dir.name}: {done or 'nichts zu tun'}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("backfill", help="bring old runs into the shared layout")
    cmd.add_argument("study_root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "backfill":
        backfill(args.study_root)


if __name__ == "__main__":
    main()
