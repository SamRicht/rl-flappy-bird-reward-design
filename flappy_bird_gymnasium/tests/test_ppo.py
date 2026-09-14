"""Correctness tests for the PPO building blocks.

The GAE recursion and the handling of episode boundaries are the parts where a
silent sign or off-by-one error would not crash but merely make learning worse,
so they are checked against an independent, deliberately naive implementation.
"""

import numpy as np
import torch

from flappy_bird_gymnasium.rl.ppo import ActorCritic, PPOConfig, RolloutBuffer


def reference_gae(
    rewards,
    values,
    dones,
    last_value,
    last_done,
    gamma,
    gae_lambda,
    truncation_values=None,
):
    """Straightforward, slow GAE for a single environment.

    Mirrors the definition
    ``A_t = sum_l (gamma * lambda)^l * delta_{t+l}`` with
    ``delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)``,
    written without any of the vectorised bookkeeping of the real buffer.
    """
    n = len(rewards)
    truncation_values = (
        truncation_values if truncation_values is not None else [0.0] * n
    )
    deltas = []
    for t in range(n):
        if t == n - 1:
            next_value, next_non_terminal = last_value, 1.0 - last_done
        else:
            next_value, next_non_terminal = values[t + 1], 1.0 - dones[t + 1]
        reward = rewards[t] + gamma * truncation_values[t]
        deltas.append(reward + gamma * next_value * next_non_terminal - values[t])

    advantages = [0.0] * n
    for t in reversed(range(n)):
        if t == n - 1:
            next_non_terminal = 1.0 - last_done
            carry = 0.0
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            carry = advantages[t + 1]
        advantages[t] = deltas[t] + gamma * gae_lambda * next_non_terminal * carry
    return advantages


def _fill_buffer(buffer, rewards, values, dones, truncation_values=None):
    buffer.rewards[:, 0] = torch.tensor(rewards, dtype=torch.float32)
    buffer.values[:, 0] = torch.tensor(values, dtype=torch.float32)
    buffer.dones[:, 0] = torch.tensor(dones, dtype=torch.float32)
    if truncation_values is not None:
        buffer.truncation_values[:, 0] = torch.tensor(
            truncation_values, dtype=torch.float32
        )


def test_gae_matches_reference_without_episode_boundary():
    device = torch.device("cpu")
    buffer = RolloutBuffer(n_steps=6, n_envs=1, obs_dim=3, device=device)
    rewards = [0.1, 0.1, 1.0, 0.1, 0.1, 0.1]
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dones = [0.0] * 6
    _fill_buffer(buffer, rewards, values, dones)

    advantages, returns = buffer.compute_advantages(
        torch.tensor([1.1]), torch.tensor([0.0]), gamma=0.99, gae_lambda=0.95
    )
    expected = reference_gae(rewards, values, dones, 1.1, 0.0, 0.99, 0.95)

    np.testing.assert_allclose(advantages[:, 0].numpy(), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        returns[:, 0].numpy(), np.array(expected) + np.array(values), rtol=1e-5
    )


def test_gae_cuts_the_trace_at_a_termination():
    device = torch.device("cpu")
    buffer = RolloutBuffer(n_steps=6, n_envs=1, obs_dim=3, device=device)
    rewards = [0.1, -1.0, 0.1, 0.1, 0.1, 0.1]
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # The episode ended on step 1, so step 2 starts a fresh one.
    dones = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    _fill_buffer(buffer, rewards, values, dones)

    advantages, _ = buffer.compute_advantages(
        torch.tensor([1.1]), torch.tensor([0.0]), gamma=0.99, gae_lambda=0.95
    )
    expected = reference_gae(rewards, values, dones, 1.1, 0.0, 0.99, 0.95)
    np.testing.assert_allclose(advantages[:, 0].numpy(), expected, rtol=1e-5, atol=1e-6)

    # The advantage of the terminal step must not depend on anything that
    # happened in the next episode.
    buffer_alt = RolloutBuffer(n_steps=6, n_envs=1, obs_dim=3, device=device)
    rewards_alt = list(rewards)
    rewards_alt[2:] = [99.0, 99.0, 99.0, 99.0]
    _fill_buffer(buffer_alt, rewards_alt, values, dones)
    advantages_alt, _ = buffer_alt.compute_advantages(
        torch.tensor([1.1]), torch.tensor([0.0]), gamma=0.99, gae_lambda=0.95
    )
    np.testing.assert_allclose(
        advantages[:2, 0].numpy(), advantages_alt[:2, 0].numpy(), rtol=1e-6
    )


def test_truncation_is_bootstrapped_not_treated_as_death():
    """A truncated step must be worth more than a terminated one."""
    device = torch.device("cpu")
    gamma, lam = 0.99, 0.95
    rewards = [0.1, 0.1, 0.1, 0.1]
    values = [0.5, 0.6, 0.7, 0.8]
    dones = [0.0, 0.0, 1.0, 0.0]  # episode ended on step 1

    terminated = RolloutBuffer(4, 1, 3, device)
    _fill_buffer(terminated, rewards, values, dones)
    adv_term, _ = terminated.compute_advantages(
        torch.tensor([0.9]), torch.tensor([0.0]), gamma, lam
    )

    truncated = RolloutBuffer(4, 1, 3, device)
    # Same boundary, but the successor state had value 5.0.
    _fill_buffer(truncated, rewards, values, dones, [0.0, 5.0, 0.0, 0.0])
    adv_trunc, _ = truncated.compute_advantages(
        torch.tensor([0.9]), torch.tensor([0.0]), gamma, lam
    )

    assert adv_trunc[1, 0] > adv_term[1, 0]
    np.testing.assert_allclose(
        (adv_trunc[1, 0] - adv_term[1, 0]).item(), gamma * 5.0, rtol=1e-5
    )
    # Steps after the boundary are unaffected.
    np.testing.assert_allclose(
        adv_trunc[2:, 0].numpy(), adv_term[2:, 0].numpy(), rtol=1e-6
    )


def test_gae_lambda_one_gives_monte_carlo_returns():
    """With lambda = 1 the value targets are the plain discounted returns."""
    device = torch.device("cpu")
    gamma = 0.9
    rewards = [1.0, 2.0, 3.0]
    values = [0.4, -0.2, 5.0]  # arbitrary; must cancel out
    buffer = RolloutBuffer(3, 1, 2, device)
    _fill_buffer(buffer, rewards, values, [0.0, 0.0, 0.0])

    _, returns = buffer.compute_advantages(
        torch.tensor([0.0]), torch.tensor([1.0]), gamma, gae_lambda=1.0
    )
    expected = [
        1.0 + gamma * 2.0 + gamma**2 * 3.0,
        2.0 + gamma * 3.0,
        3.0,
    ]
    np.testing.assert_allclose(returns[:, 0].numpy(), expected, rtol=1e-5)


def test_actor_critic_initial_policy_is_near_uniform():
    """The small policy-head gain must keep the starting policy unopinionated."""
    torch.manual_seed(0)
    model = ActorCritic(12, 2)
    obs = torch.randn(512, 12)
    with torch.no_grad():
        _, log_prob, entropy, _ = model.get_action_and_value(obs)
    # ln(2) = 0.693 is the entropy of a uniform distribution over two actions.
    assert entropy.mean().item() > 0.69
    assert log_prob.exp().mean().item() == __import__("pytest").approx(0.5, abs=0.02)


def test_separate_and_shared_backbones_produce_the_same_shapes():
    obs = torch.randn(8, 12)
    for shared in (False, True):
        model = ActorCritic(12, 2, hidden_sizes=(32, 32), shared_backbone=shared)
        action, log_prob, entropy, value = model.get_action_and_value(obs)
        assert action.shape == (8,)
        assert log_prob.shape == (8,)
        assert entropy.shape == (8,)
        assert value.shape == (8,)
        assert model.get_value(obs).shape == (8,)


def test_config_rejects_indivisible_batch():
    import pytest

    with pytest.raises(ValueError):
        PPOConfig(n_envs=8, n_steps=256, n_minibatches=7)


def test_config_round_trips_through_json():
    import json

    config = PPOConfig(hidden_sizes=(128, 64), seed=3, lr=1e-3)
    restored = PPOConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
