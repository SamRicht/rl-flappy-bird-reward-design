"""Tests for the DQN agent: n-step returns, replay buffer and the learning step.

The n-step accumulator is the part most likely to break silently -- a wrong
horizon or a missed flush costs learning speed without ever raising an error --
so its arithmetic is pinned down here with a discount that makes the expected
values easy to verify by hand.
"""

import numpy as np
import pytest

from flappy_bird_gymnasium.dqn.agent import DQNAgent
from flappy_bird_gymnasium.dqn.config import DQNConfig
from flappy_bird_gymnasium.dqn.env_utils import make_env
from flappy_bird_gymnasium.dqn.replay_buffer import NStepAccumulator, ReplayBuffer

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
