"""Turning the continuous observation into a finite state index.

A tabular method needs a finite state space, so the 12-dimensional observation
has to be reduced and binned. This is the one piece of the Q-learning work that
is actually new; the network-based methods feed the raw vector into a network
and never have to deal with it.

The reduction keeps three quantities, in raw pixels:

``dx``
    Horizontal distance from the bird's right edge to the next pipe.
``dy``
    Vertical distance from the bird's centre to the centre of that pipe's gap.
``vel``
    The bird's vertical velocity.

The rest of the observation is redundant or nearly constant. ``rot`` is a
deterministic function of the frames since the last flap, as is ``vel`` until
it saturates, so the two carry almost the same information. The absolute pipe
and bird coordinates only matter through their difference, and the third pipe
slot is the off-screen placeholder in most frames.

Two things about the observation make a naive reduction wrong: the pipe slots
are sorted by distance and not by role, and off-screen pipes are replaced by a
placeholder row whose implied gap centre lies outside the real range. Both are
handled in :mod:`flappy_bird_gymnasium.rl.geometry`, which this module reads
the observation through. The behavioural metric in ``summarize_rollout`` uses
the same code, so the rasterisation and the measurement agree on which pipe is
the current one.

Three schemes are registered in :data:`DISCRETIZERS` and selected with
``--discretizer``.
"""

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reading the observation vector lives in rl.geometry and is shared with the
# metric in summarize_rollout, so both agree on which pipe is the current one.
# Re-exported here so `from ... import discretize` still offers everything.
from flappy_bird_gymnasium.rl.geometry import (  # noqa: F401
    PLAYER_RIGHT,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    pipes_ahead,
    player_center_y,
    player_velocity,
)

#: Largest possible ``dx``: a pipe at the right screen edge. Also used when no
#: pipe is ahead at all, so "nothing in sight" lands in the same bin as "very
#: far away", which is how it behaves.
MAX_DX = float(SCREEN_WIDTH - PLAYER_RIGHT)

# Ranges for the uniform scheme. Measured over random and heuristic play: dx
# lives in [-51, 197], dy in [-300, 242] in principle but with nearly all of
# its mass in [-90, 110]. Values outside are clamped into the outermost bins,
# so the bounds only decide where the resolution is spent.
DX_RANGE = (-52.0, 200.0)
DY_RANGE = (-160.0, 160.0)
VEL_RANGE = (-9.0, 10.0)


def _bin(value: float, edges: Sequence[float]) -> int:
    """Index of the bin ``value`` falls into.

    ``n`` edges give ``n + 1`` bins. Values outside the outermost edges land
    in the outermost bins, so no separate clipping is needed.

    ``bisect`` instead of ``np.searchsorted``: the edge arrays hold a few
    dozen values, far too few for NumPy's per-call overhead to pay off, and
    this is the hot path (three times per state index, once per env step).
    The two are equivalent on a sorted sequence.
    """
    return bisect_right(edges, value)


@dataclass(frozen=True, eq=False)
class Discretizer:
    """A rasterisation of the observation into a finite state index.

    ``eq=False`` because the fields are NumPy arrays, whose element-wise
    comparison doesn't collapse into the single bool a dataclass ``__eq__``
    needs.

    Attributes:
        name: Key under which the scheme is registered.
        dx_edges: Bin boundaries for the horizontal distance, in pixels.
        dy_edges: Bin boundaries for the vertical gap offset, in pixels.
        vel_edges: Bin boundaries for the vertical velocity.
        dy2_edges: Bin boundaries for the gap offset of the pipe after the
            next one, or ``None`` when the scheme ignores it. A scheme that
            uses it gets one extra bin for "no second pipe in sight", which
            is a different situation and not just an extreme offset.
        normalize_obs: Whether observations arrive normalised.
    """

    name: str
    dx_edges: np.ndarray
    dy_edges: np.ndarray
    vel_edges: np.ndarray
    dy2_edges: Optional[np.ndarray] = None
    normalize_obs: bool = True
    # plain-Python copies of the edges, built once. bisect on a list is
    # several times faster than np.searchsorted for arrays this small, and
    # this runs three times per env step.
    _lists: Tuple[List[float], ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        edges = [self.dx_edges, self.dy_edges, self.vel_edges]
        if self.dy2_edges is not None:
            edges.append(self.dy2_edges)
        # frozen dataclass, so assign through object.__setattr__
        object.__setattr__(self, "_lists", tuple([float(e) for e in a] for a in edges))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Number of bins per dimension."""
        dims = [
            len(self.dx_edges) + 1,
            len(self.dy_edges) + 1,
            len(self.vel_edges) + 1,
        ]
        if self.dy2_edges is not None:
            dims.append(len(self.dy2_edges) + 2)  # +1 for "no second pipe"
        return tuple(dims)

    @property
    def n_states(self) -> int:
        """Size of the state space, i.e. the number of rows of the Q-table."""
        return int(np.prod(self.shape))

    @property
    def uses_lookahead(self) -> bool:
        return self.dy2_edges is not None

    def features(self, obs: Sequence[float]) -> Tuple[float, ...]:
        """The raw quantities behind the state, in pixels.

        Kept separate from :meth:`index` because these are the numbers a
        test or a diagnostic plot wants to see; the index is just bookkeeping.

        Returns:
            ``(dx, dy, vel)``, plus ``dy2`` when the scheme looks ahead.
            ``dy2`` is ``nan`` when no second pipe is in sight.
        """
        player_center = player_center_y(obs, self.normalize_obs)
        vel = player_velocity(obs, self.normalize_obs)

        ahead = pipes_ahead(obs, self.normalize_obs)
        if ahead:
            x, gap_top, gap_bottom = ahead[0]
            dx = x - PLAYER_RIGHT
            dy = player_center - (gap_top + gap_bottom) / 2
        else:
            # can't happen with the default pipe spacing, but a defined
            # fallback beats a silently wrong answer
            dx, dy = MAX_DX, 0.0

        if not self.uses_lookahead:
            return (dx, dy, vel)

        if len(ahead) >= 2:
            _, gap_top2, gap_bottom2 = ahead[1]
            dy2 = player_center - (gap_top2 + gap_bottom2) / 2
        else:
            dy2 = math.nan
        return (dx, dy, vel, dy2)

    def index(self, obs: Sequence[float]) -> int:
        """State index of ``obs``, in ``[0, n_states)``.

        The bin indices are combined in mixed radix, like flattening a
        multi-dimensional array index.
        """
        values = self.features(obs)
        lists = self._lists
        bins = [
            _bin(values[0], lists[0]),
            _bin(values[1], lists[1]),
            _bin(values[2], lists[2]),
        ]
        if self.uses_lookahead:
            dy2 = values[3]
            if math.isnan(dy2):
                # own last bin: "no second pipe" is not an extreme offset
                bins.append(len(lists[3]) + 1)
            else:
                bins.append(_bin(dy2, lists[3]))

        index = 0
        for bin_index, size in zip(bins, self.shape):
            index = index * size + bin_index
        return index

    def describe(self) -> str:
        """Human-readable summary, stored in ``config.json`` next to each run."""
        lines = [
            f"discretizer '{self.name}': {self.n_states} states, "
            f"shape {self.shape}",
            f"  dx  {_format_edges(self.dx_edges)}",
            f"  dy  {_format_edges(self.dy_edges)}",
            f"  vel {_format_edges(self.vel_edges)}",
        ]
        if self.uses_lookahead:
            lines.append(
                f"  dy2 {_format_edges(self.dy2_edges)} + 1 bin 'no second pipe'"
            )
        return "\n".join(lines)


def _format_edges(edges: np.ndarray) -> str:
    return f"{len(edges) + 1:>3} bins, edges [{', '.join(f'{e:g}' for e in edges)}]"


def _uniform_edges(low: float, high: float, bins: int) -> np.ndarray:
    """The ``bins - 1`` interior boundaries of ``bins`` equal-width bins."""
    if bins < 1:
        raise ValueError(f"need at least one bin, got {bins}")
    return np.linspace(low, high, bins + 1)[1:-1]


def _build_uniform(config) -> Discretizer:
    """Equal-width bins over the full range of each quantity.

    The simplest thing that works, and what most Flappy Bird Q-learning
    implementations use. The bin counts are configurable.
    """
    return Discretizer(
        name="uniform",
        dx_edges=_uniform_edges(*DX_RANGE, config.dx_bins),
        dy_edges=_uniform_edges(*DY_RANGE, config.dy_bins),
        vel_edges=_uniform_edges(*VEL_RANGE, config.vel_bins),
        normalize_obs=config.normalize_obs,
    )


def _build_adaptive(config) -> Discretizer:
    """Hand-placed bins: fine where the decision flips, coarse where it doesn't.

    Far from a pipe almost any action is recoverable, while a few pixels
    around the gap centre decide the episode. Spending resolution accordingly
    gives a smaller table than the uniform scheme, with higher resolution
    where it matters.

    The negative ``dx`` bins matter: ``dx < 0`` means the bird is
    horizontally inside the pipe, which is exactly when a mistake is fatal.

    Sized to roughly two thirds of the uniform default, so that a comparison
    between the two is about the placement of the bins and not about how
    many there are.
    """
    return Discretizer(
        name="adaptive",
        # fine while the bird is inside or just in front of a pipe, coarse
        # while it's still approaching
        dx_edges=np.array(
            [
                -45,
                -38,
                -31,
                -24,
                -17,
                -10,
                -3,
                4,
                11,
                18,
                25,
                32,
                40,
                50,
                62,
                76,
                92,
                110,
                132,
                158,
            ],
            dtype=float,
        ),
        # fine around the gap centre, where a few pixels decide the episode
        dy_edges=np.array(
            [
                -120,
                -95,
                -75,
                -60,
                -48,
                -38,
                -30,
                -23,
                -17,
                -12,
                -7,
                -2,
                3,
                8,
                13,
                19,
                26,
                34,
                44,
                56,
                72,
                95,
                125,
            ],
            dtype=float,
        ),
        # vel_y takes exactly 20 integer values, so 20 bins resolve it exactly
        vel_edges=_uniform_edges(*VEL_RANGE, 20),
        normalize_obs=config.normalize_obs,
    )


def _build_lookahead(config) -> Discretizer:
    """``adaptive`` plus a coarse view of the pipe after the next one.

    Lets the policy start climbing or sinking before it has passed the
    current gap. Costs a factor of five in table size, and therefore in the
    number of samples needed to fill it.
    """
    base = _build_adaptive(config)
    return Discretizer(
        name="lookahead",
        dx_edges=base.dx_edges,
        dy_edges=base.dy_edges,
        vel_edges=base.vel_edges,
        dy2_edges=np.array([-40.0, 0.0, 40.0]),
        normalize_obs=config.normalize_obs,
    )


#: The registered rasterisation schemes. ``add_config_arguments`` derives the
#: choices of ``--discretizer`` from this dict, so a new scheme is available
#: to every entry point as soon as it's registered here.
DISCRETIZERS: Dict[str, Callable[..., Discretizer]] = {
    "uniform": _build_uniform,
    "adaptive": _build_adaptive,
    "lookahead": _build_lookahead,
}


def build_discretizer(config) -> Discretizer:
    """Builds the discretiser named by ``config.discretizer``.

    Args:
        config: Anything with ``discretizer``, ``normalize_obs`` and, for the
            uniform scheme, ``dx_bins``, ``dy_bins`` and ``vel_bins``.

    Raises:
        KeyError: If the name is not registered.
    """
    if config.discretizer not in DISCRETIZERS:
        raise KeyError(
            f"unknown discretizer {config.discretizer!r}; "
            f"available: {sorted(DISCRETIZERS)}"
        )
    return DISCRETIZERS[config.discretizer](config)
