"""Reading the 12-feature observation: where is the bird relative to the gap?

The observation is a flat vector of twelve numbers with two traps in its
layout, and every piece of code that interprets it has to handle both. Keeping
that knowledge here means the metric in ``summarize_rollout`` and the state
rasterisation of the tabular agent get the same answer. They have to agree,
since one measures what the other optimises.

Layout, with the ``normalize_obs=True`` scaling in brackets::

    0,1,2   pipe slot A: x [/288], gap top y [/512], gap bottom y [/512]
    3,4,5   pipe slot B: the same triple
    6,7,8   pipe slot C: the same triple
    9       bird y, its TOP edge [/512]
    10      bird vertical velocity [/10]
    11      bird rotation [/90]

Trap 1: the three slots are sorted by x, not by role. For roughly two frames
after the bird scores, slot 0 still holds the pipe it just passed, so there is
no fixed "next pipe" index.

Trap 2: a pipe past the right screen edge is replaced by the placeholder row
``(screen_width, 0, screen_height)``, whose implied gap centre is 256, outside
the real range of ``[150, 220]``. At the start of an episode two placeholders
sit at exactly the same x as the single real pipe.

Everything here works in raw pixels, also when the observation arrives
normalised. Pixels are what the environment's own geometry uses and what a
reader of a report can picture.
"""

from typing import List, Optional, Sequence, Tuple

from flappy_bird_gymnasium.envs.constants import (
    BACKGROUND_HEIGHT,
    BACKGROUND_WIDTH,
    PIPE_HEIGHT,
    PIPE_WIDTH,
    PLAYER_HEIGHT,
    PLAYER_MAX_VEL_Y,
    PLAYER_WIDTH,
)

#: The environment's default screen size. ``make_env`` never overrides it, and
#: the observation is normalised by exactly these numbers.
SCREEN_WIDTH = BACKGROUND_WIDTH
SCREEN_HEIGHT = BACKGROUND_HEIGHT

#: The bird's horizontal position is fixed; only the pipes move.
PLAYER_X = int(SCREEN_WIDTH * 0.2)
PLAYER_RIGHT = PLAYER_X + PLAYER_WIDTH

#: Length of the feature observation, as opposed to the 180-ray LIDAR one.
FEATURE_OBS_SIZE = 12

#: Observation indices of the three pipe slots, each ``(x, gap_top, gap_bottom)``.
PIPE_SLOTS = (0, 3, 6)
#: Observation index of the bird's vertical position (its top edge).
PLAYER_Y_INDEX = 9
#: Observation index of the bird's vertical velocity.
PLAYER_VEL_INDEX = 10


def player_center_y(obs: Sequence[float], normalize_obs: bool) -> float:
    """The bird's vertical centre in pixels.

    ``obs[9]`` is the bird's top edge. The environment's own shaping potential
    compares ``player_y + PLAYER_HEIGHT / 2`` against the gap centre, so
    everything here does the same. Otherwise every offset would carry a
    12-pixel bias.
    """
    scale = SCREEN_HEIGHT if normalize_obs else 1.0
    return float(obs[PLAYER_Y_INDEX]) * scale + PLAYER_HEIGHT / 2


def player_velocity(obs: Sequence[float], normalize_obs: bool) -> float:
    """The bird's vertical velocity, in pixels per frame.

    Normalised by ``PLAYER_MAX_VEL_Y``, not by the screen height. Mixing the
    two up gives plausible small numbers instead of an error.
    """
    scale = PLAYER_MAX_VEL_Y if normalize_obs else 1.0
    return float(obs[PLAYER_VEL_INDEX]) * scale


def pipes_ahead(
    obs: Sequence[float], normalize_obs: bool
) -> List[Tuple[float, float, float]]:
    """The pipes still ahead of the bird, nearest first, in raw pixels.

    Placeholder rows are dropped. A pipe counts as "ahead" while its right
    edge hasn't passed the bird's right edge, which is the same test the
    environment uses for its shaping potential, so metric and reward agree on
    which pipe is the current one.

    Args:
        obs: A 12-feature observation (``use_lidar=False``).
        normalize_obs: Whether ``obs`` is normalised, i.e. whether the values
            have to be scaled back to pixels first.

    Returns:
        ``(x, gap_top, gap_bottom)`` per pipe, sorted by increasing ``x``.
        Possibly empty.
    """
    x_scale = SCREEN_WIDTH if normalize_obs else 1.0
    y_scale = SCREEN_HEIGHT if normalize_obs else 1.0

    ahead: List[Tuple[float, float, float]] = []
    for slot in PIPE_SLOTS:
        x = float(obs[slot]) * x_scale
        gap_top = float(obs[slot + 1]) * y_scale
        gap_bottom = float(obs[slot + 2]) * y_scale
        # the placeholder spans the whole screen height, a real pipe never does
        if gap_top <= 0.0 and gap_bottom >= SCREEN_HEIGHT:
            continue
        if x + PIPE_WIDTH < PLAYER_RIGHT:
            continue
        ahead.append((x, gap_top, gap_bottom))

    ahead.sort(key=lambda pipe: pipe[0])
    return ahead


def gap_offset(obs: Sequence[float], normalize_obs: bool = True) -> Optional[float]:
    """Signed vertical distance from the bird's centre to the gap it's flying at.

    A behavioural measure, not a performance one: two reward schemes can
    reach the same score while one hugs the centre of the gap and the other
    skims its edge. That difference is what a study of reward design is
    after, and the score alone doesn't show it.

    Args:
        obs: A 12-feature observation. Anything else returns ``None``.
        normalize_obs: Whether ``obs`` is normalised.

    Returns:
        Pixels, positive when the bird is below the centre of the gap, or
        ``None`` when no pipe lies ahead.

    Note:
        Not the same quantity as ``_gap_offset`` in the PPO work on branch
        ``fb_drl_ppo``, which measures from the bird's top edge and reports
        screen heights. The two differ by a constant ``PLAYER_HEIGHT / 2 = 12``
        pixels plus the factor 512, so don't put their numbers in one table
        without converting first.
    """
    if len(obs) != FEATURE_OBS_SIZE:
        return None

    ahead = pipes_ahead(obs, normalize_obs)
    if not ahead:
        return None

    _, gap_top, gap_bottom = ahead[0]
    return player_center_y(obs, normalize_obs) - (gap_top + gap_bottom) / 2


def env_gap_offset(env) -> Optional[float]:
    """:func:`gap_offset`, read off the environment's state instead of the obs.

    Works under every observation type: the 12 features, LIDAR and the stacked
    pixel frames of the CNN work all leave the game state the same, while only
    the first carries the pipe positions in the observation. The same two
    rules as in :func:`pipes_ahead` apply, so under feature observations both
    functions agree to the pixel (checked frame by frame in the tests):

    * a pipe past the right screen edge doesn't count, it is the placeholder
      row in the observation;
    * the target is the nearest pipe whose right edge hasn't passed the
      bird's right edge.

    Reads private attributes of ``FlappyBirdEnv``. That couples this function
    to the game's internals, which is the price for a measurement that doesn't
    depend on what the agent gets to see.

    Args:
        env: The environment, wrapped or not.

    Returns:
        Pixels, positive when the bird is below the centre of the gap, or
        ``None`` when no pipe lies ahead on screen.
    """
    core = env.unwrapped
    ahead = [
        (low["x"], (up["y"] + PIPE_HEIGHT + low["y"]) / 2)
        for up, low in zip(core._upper_pipes, core._lower_pipes)
        if low["x"] <= core._screen_width and low["x"] + PIPE_WIDTH >= PLAYER_RIGHT
    ]
    if not ahead:
        return None
    _, gap_centre = min(ahead, key=lambda pipe: pipe[0])
    return float(core._player_y) + PLAYER_HEIGHT / 2 - gap_centre
