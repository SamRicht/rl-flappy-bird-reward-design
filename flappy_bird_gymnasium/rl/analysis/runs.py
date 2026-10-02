"""The run-directory contract, and how a learning method plugs into it.

Every stage of the evaluation chain needs the same two things from a method:
where its runs are, and how to turn a checkpoint into a policy. The first is
fixed by the directory layout described in :mod:`flappy_bird_gymnasium.rl.analysis`.
The second is a :class:`RunLoader`, one per method, registered by name in
:data:`LOADERS`.

The registry holds import paths rather than the loaders themselves, so this
module never imports a method's package. The tabular agent needs NumPy only,
the others need PyTorch, and the CI workflow doesn't install it.
"""

import csv
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium
import numpy as np

from flappy_bird_gymnasium.rl.rollout import make_env
from flappy_bird_gymnasium.rl.runlog import THRESHOLDS, steps_to_thresholds

#: Where a run's per-episode training log lives unless its loader says
#: otherwise, and which column in it counts the environment steps.
TRAIN_LOG = "train.csv"
STEP_COLUMN = "step"


@dataclass
class LoadedRun:
    """A checkpoint, ready to be measured.

    Attributes:
        config: The run's configuration. Duck-typed like the configs in
            :mod:`flappy_bird_gymnasium.rl.rollout`, plus ``seed`` and
            ``eval_max_episode_steps``.
        policy: Maps an observation to an action, greedily. Measuring an
            exploring policy would mix what the agent learned with how much
            it still explores.
        columns: Method-specific values written next to the shared metrics,
            e.g. the table size of the tabular agent. Empty is fine.
        make_env: Builds the environment this policy plays in, for a method
            whose observations differ from the default ones (the CNN work's
            stacked pixel frames). Called as ``make_env(max_episode_steps=...,
            render_mode=...)``; build it with
            :func:`flappy_bird_gymnasium.rl.rollout.make_env` and its
            ``wrapper`` argument, so reward scheme and frame limit stay the
            shared ones. ``None`` means the default environment.
    """

    config: Any
    policy: Callable[[np.ndarray], int]
    columns: Dict[str, object] = field(default_factory=dict)
    make_env: Optional[Callable[..., gymnasium.Env]] = None


@dataclass(frozen=True)
class RunLoader:
    """How the evaluation chain reads the runs of one learning method.

    Attributes:
        algorithm: Written into every result row, so the rows of several
            methods can be concatenated into one table.
        load: Turns a checkpoint path into a :class:`LoadedRun`. Has to be a
            module-level function, it is sent to worker processes.
        checkpoints: File names of the checkpoints a run holds, the default
            first.
        metrics: Numeric entries of :attr:`LoadedRun.columns` that differ
            between seeds and are aggregated like the shared metrics.
        train_log: The per-episode training log inside a run directory, at
            least a step and a ``score`` column. The PPO work writes
            ``episodes.csv``.
        step_column: The column of ``train_log`` holding the environment
            step; ``global_step`` in the PPO work.
    """

    algorithm: str
    load: Callable[[Path], LoadedRun]
    checkpoints: Tuple[str, ...]
    metrics: Tuple[str, ...] = ()
    train_log: str = TRAIN_LOG
    step_column: str = STEP_COLUMN

    @property
    def default_checkpoint(self) -> str:
        return self.checkpoints[0]


#: Every method the chain can measure, as ``"module:attribute"``. Another
#: method joins by adding its line here and a ``RunLoader`` in its package.
LOADERS: Dict[str, str] = {
    "qlearning": "flappy_bird_gymnasium.qlearning.evaluate:RUN_LOADER",
    "cnn": "flappy_bird_gymnasium.evaluate_cnn:RUN_LOADER",
}


def resolve_loader(name: str) -> RunLoader:
    """The loader registered as ``name``, or one given as ``"module:attribute"``.

    The second form lets a method use the chain before it is registered.
    """
    target = LOADERS.get(name, name)
    if ":" not in target:
        raise SystemExit(
            f"unknown algorithm {name!r}; registered: {sorted(LOADERS)}, "
            "or pass 'package.module:ATTRIBUTE'"
        )
    module_name, attribute = target.split(":", 1)
    loader = getattr(importlib.import_module(module_name), attribute)
    if not isinstance(loader, RunLoader):
        raise SystemExit(f"{target} is not a RunLoader")
    return loader


def open_env(
    loaded: LoadedRun,
    max_episode_steps: int,
    render_mode: Optional[str] = None,
) -> gymnasium.Env:
    """The environment ``loaded`` plays in: its own, or the default one."""
    if loaded.make_env is not None:
        return loaded.make_env(
            max_episode_steps=max_episode_steps, render_mode=render_mode
        )
    return make_env(
        loaded.config, render_mode=render_mode, max_episode_steps=max_episode_steps
    )


def variant_of(run_dir: Path) -> str:
    """``reward_legacy_seed3`` -> ``reward_legacy``."""
    return run_dir.name.rsplit("_seed", 1)[0]


def collect_runs(root: Path) -> Dict[str, List[Path]]:
    """Groups the finished runs under ``root`` by variant.

    The grouping comes from the directory names (``<variant>_seed<n>``), not
    from an index file, so a study stays readable even if it was interrupted
    or assembled by hand. A run counts as finished once it wrote its
    ``summary.json``, which happens last.
    """
    grouped: Dict[str, List[Path]] = {}
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "summary.json").exists():
            grouped.setdefault(variant_of(path), []).append(path)
    return grouped


def unfinished_runs(root: Path) -> List[Path]:
    """Runs that started but never wrote their summary.

    Worth reporting rather than skipping quietly: an interrupted study
    compares variants on different numbers of seeds, and nothing in the
    result table would show it otherwise.
    """
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and (path / "config.json").exists()
        and not (path / "summary.json").exists()
    ]


def read_column(
    run_dir: Path,
    column: str,
    source: str = TRAIN_LOG,
    step_column: str = STEP_COLUMN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reads ``(step, value)`` from one of a run's CSV logs.

    Empty and unparsable cells are skipped together with their step, so the
    two arrays always stay aligned. A missing file or column gives two empty
    arrays, since not every method logs every diagnostic.
    """
    steps: List[float] = []
    values: List[float] = []
    path = run_dir / source
    if not path.exists():
        return np.array([]), np.array([])
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if step_column not in (reader.fieldnames or []):
            raise SystemExit(
                f"{path} has no step column {step_column!r}; its columns are "
                f"{reader.fieldnames}. Pass the right one (PPO: global_step)."
            )
        for row in reader:
            raw = row.get(column, "")
            if raw in ("", None):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            values.append(value)
            steps.append(float(row[step_column]))
    return np.asarray(steps), np.asarray(values)


def sample_efficiency(
    run_dir: Path,
    thresholds: Sequence[int] = THRESHOLDS,
    source: str = TRAIN_LOG,
    step_column: str = STEP_COLUMN,
) -> Dict[str, Optional[int]]:
    """``steps_to_<n>`` of one run, recomputed from its training log.

    Recomputed rather than read from ``summary.json``, so every run of a
    comparison is measured against the same thresholds, including runs
    trained before the thresholds last changed. Falls back to the summary
    for a run without a usable training log.
    """
    steps, scores = read_column(run_dir, "score", source, step_column)
    if steps.size:
        return steps_to_thresholds(
            steps.astype(int).tolist(), scores.tolist(), tuple(thresholds)
        )
    summary_path = run_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    return {f"steps_to_{t}": summary.get(f"steps_to_{t}") for t in thresholds}
