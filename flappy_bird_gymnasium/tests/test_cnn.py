"""Tests the pixel observation wrapper and the convolutional Q-network."""

import numpy as np
import pytest

import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.pixel_wrapper import make_pixel_env


def test_pixel_env_observation_shape():
    env = make_pixel_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8
    assert env.observation_space.shape == (4, 84, 84)

    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (4, 84, 84)
    # The frame must contain something (bird, pipes, ground), not a flat image.
    assert obs[-1].min() != obs[-1].max()
    env.close()


def test_cnn_forward_and_action():
    torch = pytest.importorskip("torch")
    from flappy_bird_gymnasium.tests.cnn_dqn import DuelingCNN

    env = make_pixel_env()
    obs, _ = env.reset(seed=0)
    model = DuelingCNN(int(env.action_space.n))

    q_values = model(torch.as_tensor(obs).unsqueeze(0))
    assert q_values.shape == (1, 2)
    assert torch.isfinite(q_values).all()

    action = model.get_action(obs)
    assert action in (0, 1)
    env.close()
