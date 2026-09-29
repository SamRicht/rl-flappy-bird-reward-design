"""Checks that a PPO run still satisfies the shared evaluation chain's contract.

The contract is duck-typed on both sides: the chain reads attributes off a
config object it never declared, and the loader fills them from a ``config.json``
whose layout is the training's business.  Nothing connects the two but these
tests, so a renamed field would surface as a measurement that silently used a
default instead of failing.
"""

import json

import pytest

from flappy_bird_gymnasium.rl.analysis.runs import LoadedRun, RunLoader
from flappy_bird_gymnasium.rl.envs import EnvConfig
from flappy_bird_gymnasium.rl.ppo import PPOConfig
from flappy_bird_gymnasium.rl.ppo_loader import RUN_LOADER, PPORunConfig, load
from flappy_bird_gymnasium.rl.rewards import PRESETS
from flappy_bird_gymnasium.rl.rollout import make_env, reward_config_for
from flappy_bird_gymnasium.rl.train import train


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A real, tiny training run -- the fixture has to be genuine output.

    A hand-written ``config.json`` would keep passing after the training stopped
    writing the fields the loader reads, which is the failure this guards.
    """
    directory = tmp_path_factory.mktemp("ppo_run") / "risk_averse_seed3"
    train(
        PPOConfig(total_steps=4096, n_envs=2, n_steps=256, seed=3),
        PRESETS["risk_averse"],
        EnvConfig(max_episode_steps=500),
        directory,
        verbose=False,
    )
    return directory


def test_loader_is_registrable():
    assert isinstance(RUN_LOADER, RunLoader)
    assert RUN_LOADER.algorithm == "ppo"
    # The chain sends `load` to worker processes, which pickle it by name.
    assert RUN_LOADER.load.__module__ and RUN_LOADER.load.__qualname__ == "load"


def test_training_writes_what_the_chain_looks_for(run_dir):
    assert (run_dir / "summary.json").exists(), "marks the run as finished"
    assert (run_dir / "config.json").exists()
    assert (run_dir / RUN_LOADER.train_log).exists()
    log = (run_dir / RUN_LOADER.train_log).read_text(encoding="utf-8")
    columns = log.splitlines()[0].split(",")
    assert RUN_LOADER.step_column in columns
    assert "score" in columns
    assert (run_dir / RUN_LOADER.default_checkpoint).exists()


def test_loaded_config_carries_every_attribute_the_chain_reads(run_dir):
    loaded = load(run_dir / RUN_LOADER.default_checkpoint)
    assert isinstance(loaded, LoadedRun)
    for attribute in (
        "use_lidar",
        "normalize_obs",
        "pipe_gap",
        "reward_preset",
        "reward_overrides",
        "gamma",
        "max_episode_steps",
        "seed",
        "eval_max_episode_steps",
    ):
        assert hasattr(loaded.config, attribute), attribute


def test_the_reward_is_reconstructed_exactly(run_dir):
    """The measured policy has to play under the reward it was trained on."""
    trained = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rebuilt = reward_config_for(load(run_dir / "model.pt").config)
    assert rebuilt.to_dict() == trained["reward"]


def test_the_policy_plays_the_game(run_dir):
    loaded = load(run_dir / "model.pt")
    env = make_env(loaded.config, max_episode_steps=200)
    obs, _ = env.reset(seed=0)
    action = loaded.policy(obs)
    assert action in (0, 1)
    env.step(action)
    env.close()


def test_the_policy_is_greedy(run_dir):
    """Measuring an exploring policy would mix learning with leftover noise."""
    loaded = load(run_dir / "model.pt")
    env = make_env(loaded.config, max_episode_steps=200)
    obs, _ = env.reset(seed=0)
    env.close()
    assert len({loaded.policy(obs) for _ in range(30)}) == 1


def test_config_json_has_not_grown_a_second_reward_name(run_dir):
    """`reward_overrides` must not carry `name`, which is not a reward term."""
    config = load(run_dir / "model.pt").config
    assert isinstance(config, PPORunConfig)
    assert "name" not in config.reward_overrides
