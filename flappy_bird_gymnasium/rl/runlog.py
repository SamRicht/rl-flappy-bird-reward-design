"""Run bookkeeping shared by every learning method: CSV logs and metrics.

Kept in one place so the output of two agents is comparable without checking
column by column whether "score" means the same thing on both sides.
``eval.csv`` in particular is written from :data:`EVAL_LOG_FIELDS` and filled
from :func:`flappy_bird_gymnasium.rl.rollout.summarize_rollout`, so the file
format can't drift from what is actually measured.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Columns of ``eval.csv``: the training step plus everything
#: ``summarize_rollout`` reports, in that order.
EVAL_LOG_FIELDS: List[str] = [
    "step",
    "mean_return",
    "mean_score",
    "median_score",
    "std_score",
    "max_score",
    "min_score",
    "mean_length",
    "mean_flap_rate",
    "truncation_rate",
    "mean_gap_offset",
    "mean_abs_gap_offset",
]

#: Score levels for the sample-efficiency metric. Unlike the final score,
#: "steps until the policy sustains X" can't be censored by the episode limit,
#: so it keeps separating variants after they have all saturated. It also stays
#: meaningful across methods, where the absolute score doesn't.
THRESHOLDS: Tuple[int, ...] = (1, 5, 10, 25, 50)


class CsvLogger:
    """Appends rows to a CSV file, keeping the handle open for the whole run.

    Flushes after every row so that a killed run still leaves a readable log,
    without reopening the file thousands of times.
    """

    def __init__(self, path: Path, fieldnames: Sequence[str]):
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=list(fieldnames))
        self._writer.writeheader()

    def log(self, row: Dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def steps_to_thresholds(
    steps: Sequence[int],
    scores: Sequence[float],
    thresholds: Tuple[int, ...] = THRESHOLDS,
    window: int = 20,
) -> Dict[str, Optional[int]]:
    """Environment steps until the rolling mean score first reaches each level.

    The metric that survives the episode limit: it measures how fast a
    variant learned, which stays meaningful after every variant has saturated
    at the capped score. The rolling mean keeps a single lucky episode from
    counting as "reached".

    Args:
        steps: Environment step at which each episode ended.
        scores: Score of each episode, same length and order as ``steps``.
        thresholds: Score levels to report.
        window: Number of episodes in the rolling mean.

    Returns:
        ``{"steps_to_<level>": step or None}``. ``None`` means the level was
        never reached and has to stay ``None``; reading it as 0 would make
        the worst result look like the best.
    """
    result: Dict[str, Optional[int]] = {f"steps_to_{t}": None for t in thresholds}
    if len(scores) < window:
        return result

    rolling = np.convolve(
        np.asarray(scores, dtype=float), np.ones(window) / window, "valid"
    )
    # rolling[i] covers episodes i .. i+window-1, credited to the last one.
    # Named offset because black formats the slice as `steps[window - 1 :]`
    # and flake8 then reports E203.
    first_complete = window - 1
    rolling_steps = np.asarray(steps[first_complete:])
    for threshold in thresholds:
        hit = np.flatnonzero(rolling >= threshold)
        if hit.size:
            result[f"steps_to_{threshold}"] = int(rolling_steps[hit[0]])
    return result
