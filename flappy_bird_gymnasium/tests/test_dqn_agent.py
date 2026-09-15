"""Tests for the DQN agent: n-step returns, replay buffer and the learning step.

The n-step accumulator is the part most likely to break silently -- a wrong
horizon or a missed flush costs learning speed without ever raising an error --
so its arithmetic is pinned down here with a discount that makes the expected
values easy to verify by hand.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from flappy_bird_gymnasium.dqn.agent import DQNAgent
from flappy_bird_gymnasium.dqn.config import DQNConfig
from flappy_bird_gymnasium.dqn.env_utils import make_env
from flappy_bird_gymnasium.dqn.experiments import ABLATIONS, SWEEPABLE, _parse_hidden
from flappy_bird_gymnasium.dqn.replay_buffer import NStepAccumulator, ReplayBuffer
from flappy_bird_gymnasium.dqn.train import steps_to_thresholds

GAMMA = 0.5  # powers of two keep the expected returns exact in floating point


def _obs(value: float) -> np.ndarray:
    return np.full(3, value, dtype=np.float32)


class TestNStepAccumulator:
    def test_single_step_is_pass_through(self):
        acc = NStepAccumulator(n_step=1, gamma=GAMMA)
        (transition,) = acc.push(_obs(0), 1, 2.0, _obs(1), False)
        obs, action, reward, next_obs, terminated, discount = transition
        assert action == 1
        assert reward == pytest.approx(2.0)
        assert discount == pytest.approx(GAMMA)
        assert terminated is False
        np.testing.assert_array_equal(obs, _obs(0))
        np.testing.assert_array_equal(next_obs, _obs(1))

    def test_emits_nothing_until_the_horizon_is_full(self):
        acc = NStepAccumulator(n_step=3, gamma=GAMMA)
        assert acc.push(_obs(0), 0, 1.0, _obs(1), False) == []
        assert acc.push(_obs(1), 0, 2.0, _obs(2), False) == []
        assert len(acc.push(_obs(2), 0, 3.0, _obs(3), False)) == 1

    def test_discounted_return_over_full_horizon(self):
        acc = NStepAccumulator(n_step=3, gamma=GAMMA)
        acc.push(_obs(0), 0, 1.0, _obs(1), False)
        acc.push(_obs(1), 0, 2.0, _obs(2), False)
        (transition,) = acc.push(_obs(2), 0, 3.0, _obs(3), False)

        obs, _, reward, next_obs, terminated, discount = transition
        # 1 + 0.5*2 + 0.25*3
        assert reward == pytest.approx(2.75)
        assert discount == pytest.approx(GAMMA**3)
        assert terminated is False
        np.testing.assert_array_equal(obs, _obs(0))
        np.testing.assert_array_equal(next_obs, _obs(3))

    def test_termination_flushes_with_shrinking_horizons(self):
        acc = NStepAccumulator(n_step=3, gamma=GAMMA)
        acc.push(_obs(0), 0, 1.0, _obs(1), False)
        acc.push(_obs(1), 0, 2.0, _obs(2), False)
        acc.push(_obs(2), 0, 3.0, _obs(3), False)  # emits the first transition
        flushed = acc.push(_obs(3), 0, 4.0, _obs(4), True)

        assert len(flushed) == 3
        expected = [
            (2.0 + 0.5 * 3.0 + 0.25 * 4.0, GAMMA**3),
            (3.0 + 0.5 * 4.0, GAMMA**2),
            (4.0, GAMMA),
        ]
        for transition, (exp_reward, exp_discount) in zip(flushed, expected):
            _, _, reward, next_obs, terminated, discount = transition
            assert reward == pytest.approx(exp_reward)
            assert discount == pytest.approx(exp_discount)
            # everything still queued ended in the same terminal state
            assert terminated is True
            np.testing.assert_array_equal(next_obs, _obs(4))

    def test_truncation_keeps_the_bootstrap_alive(self):
        """A truncated episode must never be stored as a death."""
        acc = NStepAccumulator(n_step=3, gamma=GAMMA)
        acc.push(_obs(0), 0, 1.0, _obs(1), False)
        acc.push(_obs(1), 0, 2.0, _obs(2), False)

        flushed = acc.flush()
        assert len(flushed) == 2
        assert all(transition[4] is False for transition in flushed)

    def test_reset_drops_pending_steps(self):
        acc = NStepAccumulator(n_step=3, gamma=GAMMA)
        acc.push(_obs(0), 0, 1.0, _obs(1), False)
        acc.reset()
        assert acc.flush() == []

    def test_rejects_invalid_horizon(self):
        with pytest.raises(ValueError):
            NStepAccumulator(n_step=0, gamma=GAMMA)


class TestReplayBuffer:
    def test_grows_then_overwrites_oldest(self):
        buffer = ReplayBuffer(capacity=3, obs_dim=3, seed=0)
        for i in range(5):
            buffer.add(_obs(i), i % 2, float(i), _obs(i + 1), False, 0.9)
        assert len(buffer) == 3
        # the two oldest entries were overwritten by the ring
        assert set(buffer.rewards.tolist()) == {2.0, 3.0, 4.0}

    def test_sample_returns_aligned_batch(self):
        buffer = ReplayBuffer(capacity=10, obs_dim=3, seed=0)
        for i in range(10):
            buffer.add(_obs(i), 1, float(i), _obs(i + 1), i == 9, 0.5)

        obs, actions, rewards, next_obs, dones, discounts = buffer.sample(4)
        assert obs.shape == (4, 3)
        assert next_obs.shape == (4, 3)
        for array in (actions, rewards, dones, discounts):
            assert array.shape == (4,)
        # obs[i] and reward[i] must describe the same transition
        np.testing.assert_allclose(obs[:, 0], rewards)


class TestDQNAgent:
    def _agent(self, **overrides) -> DQNAgent:
        config = DQNConfig(
            buffer_size=500,
            batch_size=8,
            learning_starts=16,
            n_step=3,
            seed=0,
            **overrides,
        )
        return DQNAgent(obs_dim=3, n_actions=2, config=config)

    def test_learn_waits_for_the_buffer_to_fill(self):
        agent = self._agent()
        assert agent.learn() is None

    def test_learn_reports_metrics_once_ready(self):
        agent = self._agent()
        for i in range(60):
            agent.remember(_obs(i), i % 2, 0.1, _obs(i + 1), i % 20 == 19)
        metrics = agent.learn()
        assert metrics is not None
        assert np.isfinite(metrics["loss"])
        assert np.isfinite(metrics["q_mean"])

    def test_epsilon_decays_to_its_floor(self):
        agent = self._agent(epsilon_decay_steps=1000)
        assert agent.epsilon(0) == pytest.approx(agent.cfg.epsilon_start)
        assert agent.epsilon(10_000) == pytest.approx(agent.cfg.epsilon_end)

    def test_greedy_action_is_deterministic(self):
        agent = self._agent()
        observation = _obs(0.3)
        assert agent.act(observation, epsilon=0.0) == agent.act(observation, epsilon=0.0)

    def test_save_and_load_round_trip(self, tmp_path):
        agent = self._agent()
        for i in range(60):
            agent.remember(_obs(i), i % 2, 0.1, _obs(i + 1), i % 20 == 19)
        agent.learn()

        path = tmp_path / "checkpoint.pt"
        agent.save(path, step=123)
        restored = DQNAgent.load(path)

        assert restored.cfg.n_step == agent.cfg.n_step
        assert restored.train_steps == agent.train_steps
        observation = _obs(0.42)
        assert restored.act(observation) == agent.act(observation)


class TestEnvIntegration:
    @pytest.mark.parametrize("preset", ["legacy", "sparse", "shaped", "energy"])
    def test_agent_runs_an_episode_under_every_reward_scheme(self, preset):
        config = DQNConfig(reward_preset=preset, max_episode_steps=60, seed=0)
        env = make_env(config)
        agent = DQNAgent(env.observation_space.shape[0], int(env.action_space.n), config)

        obs, _ = env.reset(seed=0)
        for _ in range(60):
            action = agent.act(obs, epsilon=0.5)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert "score" in info and "flaps" in info
        env.close()

    def test_time_limit_truncates_instead_of_terminating(self):
        """The step limit must arrive as a truncation, never as a death."""
        config = DQNConfig(max_episode_steps=5, seed=0)
        env = make_env(config)
        env.reset(seed=0)
        # action 1 (flap) keeps the bird airborne well past five frames
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step(1)
            assert not (terminated or truncated)
        _, _, terminated, truncated, _ = env.step(1)
        assert truncated and not terminated
        env.close()

    def test_measuring_uses_the_higher_limit(self):
        """Training and measuring must not share a frame limit.

        Measuring against the training limit censors every competent policy at
        the same score -- the dqn_v2 baseline reported 79 for a policy worth
        304 pipes.
        """
        config = DQNConfig(max_episode_steps=5, eval_max_episode_steps=40, seed=0)

        train_env = make_env(config)
        train_env.reset(seed=0)
        for step in range(1, 6):
            _, _, _, truncated, _ = train_env.step(1)
        assert truncated, "training env should stop at its own limit"
        train_env.close()

        eval_env = make_env(config, evaluation=True)
        eval_env.reset(seed=0)
        for _ in range(6):
            _, _, terminated, truncated, _ = eval_env.step(1)
            assert not truncated, "measuring env must run past the training limit"
            if terminated:
                break
        eval_env.close()


class TestSampleEfficiency:
    """`steps_to_thresholds` is the metric that survives the frame limit."""

    def test_reports_the_step_a_level_was_sustained(self):
        # scores climb from 0 to 4; with window=2 the rolling mean reaches
        # 1.0 once two consecutive episodes average 1.0
        steps = [100, 200, 300, 400, 500]
        scores = [0.0, 0.0, 2.0, 2.0, 4.0]
        result = steps_to_thresholds(steps, scores, thresholds=(1, 3), window=2)
        # rolling means: 0.0 (@200), 1.0 (@300), 2.0 (@400), 3.0 (@500)
        assert result["steps_to_1"] == 300
        assert result["steps_to_3"] == 500

    def test_unreached_levels_are_none_not_zero(self):
        """A level that was never reached must be absent, not a small number."""
        result = steps_to_thresholds([1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0], (5,), window=2)
        assert result["steps_to_5"] is None

    def test_too_few_episodes_yields_no_claims(self):
        result = steps_to_thresholds([1], [99.0], thresholds=(1,), window=20)
        assert result["steps_to_1"] is None

    def test_a_single_lucky_episode_does_not_count(self):
        """The rolling mean is what makes the metric robust."""
        steps = [100, 200, 300, 400]
        scores = [0.0, 50.0, 0.0, 0.0]  # one outlier
        result = steps_to_thresholds(steps, scores, thresholds=(40,), window=4)
        assert result["steps_to_40"] is None


class TestAblations:
    """Every ablation variant must produce a config the agent accepts."""

    @pytest.mark.parametrize("variant", sorted(ABLATIONS))
    def test_variant_builds_a_working_agent(self, variant):
        config = replace(
            DQNConfig(seed=0, buffer_size=200, learning_starts=16, batch_size=8),
            **ABLATIONS[variant],
        )
        agent = DQNAgent(obs_dim=3, n_actions=2, config=config)
        for i in range(40):
            agent.remember(_obs(i), i % 2, 0.1, _obs(i + 1), i % 10 == 9)
        metrics = agent.learn()
        assert metrics is not None
        assert np.isfinite(metrics["loss"])

    def test_vanilla_switches_everything_off(self):
        assert ABLATIONS["vanilla"] == {
            "double_dqn": False,
            "dueling": False,
            "n_step": 1,
        }


class TestReproducibility:
    """A seed has to mean something, or no seed comparison is worth anything."""

    def _agent(self, seed: int) -> DQNAgent:
        return DQNAgent(
            obs_dim=3,
            n_actions=2,
            config=DQNConfig(seed=seed, buffer_size=100, batch_size=4, learning_starts=8),
        )

    def test_same_seed_gives_the_same_exploration(self):
        first, second = self._agent(7), self._agent(7)
        actions_a = [first.act(_obs(i / 10), epsilon=0.5) for i in range(40)]
        actions_b = [second.act(_obs(i / 10), epsilon=0.5) for i in range(40)]
        assert actions_a == actions_b

    def test_different_seeds_diverge(self):
        first, second = self._agent(7), self._agent(8)
        actions_a = [first.act(_obs(i / 10), epsilon=0.5) for i in range(40)]
        actions_b = [second.act(_obs(i / 10), epsilon=0.5) for i in range(40)]
        assert actions_a != actions_b

    def test_same_seed_gives_the_same_replay_batch(self):
        first, second = self._agent(3), self._agent(3)
        for agent in (first, second):
            for i in range(30):
                agent.remember(_obs(i), i % 2, float(i), _obs(i + 1), i % 7 == 6)
        batch_a = first.buffer.sample(4)
        batch_b = second.buffer.sample(4)
        for array_a, array_b in zip(batch_a, batch_b):
            np.testing.assert_array_equal(array_a, array_b)


class TestSweepParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [("256x256", (256, 256)), ("64,64", (64, 64)), ("512", (512,))],
    )
    def test_hidden_sizes_parse(self, text, expected):
        assert _parse_hidden(text) == expected

    def test_hidden_is_sweepable(self):
        """Network size must be varyable like any other hyperparameter."""
        assert "hidden" in SWEEPABLE
        config = replace(DQNConfig(), hidden=SWEEPABLE["hidden"]("128x64"))
        agent = DQNAgent(obs_dim=3, n_actions=2, config=config)
        assert agent.act(_obs(0.5)) in (0, 1)


class TestConfigRoundTrip:
    def test_new_fields_survive_json(self):
        config = DQNConfig(n_step=5, eval_max_episode_steps=12_345, hidden=(64, 32))
        restored = DQNConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        assert restored.n_step == 5
        assert restored.eval_max_episode_steps == 12_345
        # tuples come back from JSON as lists and must be restored as tuples
        assert restored.hidden == (64, 32)
