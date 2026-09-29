"""Tests for the seeding of the training environments.

The seeding is worth a test of its own because a mistake in it is invisible:
nothing crashes, training still converges, and only the reported *uncertainty*
comes out wrong.  An earlier version derived the per-environment seeds as
``seed + idx``, which let consecutive runs share seven of their eight
environments, and that survived unnoticed through several studies.
"""

from flappy_bird_gymnasium.rl.envs import EnvConfig, env_seeds, make_vector_env
from flappy_bird_gymnasium.rl.rewards import PRESETS

N_ENVS = 4


def pipe_layouts(reward, run_seed, n_envs=N_ENVS):
    """The initial pipe heights of every sub-environment of one run.

    Built exactly the way ``train.py`` builds them, including the explicit seed
    list at ``reset``: that call, not the constructor, seeds the generator the
    pipes are drawn from.
    """
    envs = make_vector_env(EnvConfig(), PRESETS[reward], n_envs, seed=run_seed)
    envs.reset(seed=env_seeds(run_seed, n_envs))
    layouts = [
        tuple(pipe["y"] for pipe in sub.unwrapped._upper_pipes) for sub in envs.envs
    ]
    envs.close()
    return layouts


def test_env_seed_blocks_do_not_overlap():
    blocks = [set(env_seeds(seed, N_ENVS)) for seed in range(6)]
    for i, first in enumerate(blocks):
        assert len(first) == N_ENVS
        for second in blocks[i + 1 :]:
            assert not first & second


def test_consecutive_runs_do_not_share_environments():
    """The failure this guards against: run 1 replaying run 0's episodes."""
    for a, b in ((0, 1), (1, 2), (5, 6)):
        first = pipe_layouts("legacy", a)
        second = pipe_layouts("legacy", b)
        # A stream reused verbatim would show up as an identical layout at some
        # position; equal layouts by chance are possible but not at the same
        # index for a whole run.
        assert first != second
        assert not any(x == y for x, y in zip(first, second))


def test_reward_designs_at_the_same_seed_see_the_same_game():
    """The comparison is paired: only the reward may differ between variants."""
    for seed in (0, 3):
        reference = pipe_layouts("legacy", seed)
        for reward in ("risk_averse", "energy", "sparse"):
            assert pipe_layouts(reward, seed) == reference


def test_a_plain_int_at_reset_would_reintroduce_the_overlap():
    """Documents *why* the seed list is passed explicitly.

    Gymnasium expands ``reset(seed=n)`` into ``[n, n + 1, ...]``.  If this test
    ever fails, Gymnasium changed that behaviour and the comment in ``train.py``
    is out of date.
    """
    envs = make_vector_env(EnvConfig(), PRESETS["legacy"], N_ENVS, seed=0)
    envs.reset(seed=0)
    shifted = [
        tuple(pipe["y"] for pipe in sub.unwrapped._upper_pipes) for sub in envs.envs
    ]
    envs.close()

    envs = make_vector_env(EnvConfig(), PRESETS["legacy"], N_ENVS, seed=1)
    envs.reset(seed=1)
    shifted_next = [
        tuple(pipe["y"] for pipe in sub.unwrapped._upper_pipes) for sub in envs.envs
    ]
    envs.close()

    assert shifted[1:] == shifted_next[:-1]
