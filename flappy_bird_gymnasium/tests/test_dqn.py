"""Template for evaluating a self-trained DQN agent.

This file contains *no* training. It loads a checkpoint and lets the agent play
greedily. It is meant as a scaffold for evaluating the reward variants: same
policy, same number of episodes, only the environment's reward function differs.

The pretrained models originally used here (assets/model/*.h5) were removed:
they were trained with the original reward function and are therefore useless
for comparing our own reward variants. The same applies to the transformer
variant (dueling_v2.py) and its FrameStack wrapper.
"""

import os

import gymnasium
import numpy as np

import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.utils import MODEL_PATH
from flappy_bird_gymnasium.tests.dueling import DuelingDQN

# Keras 3 requires the ".weights.h5" suffix when saving/loading weights.
DEFAULT_MODEL_FILE = os.path.join(MODEL_PATH, "dqn.weights.h5")


def build_model(env):
    """Builds the Q-network matching the environment and creates its weights."""
    # int() is required: gymnasium returns action_space.n as numpy.int64, and
    # Keras 3 checks `isinstance(units, int)` strictly, rejecting numpy types.
    q_model = DuelingDQN(int(env.action_space.n))
    q_model.build((None, *env.observation_space.shape))
    return q_model


def play(
    model_file=DEFAULT_MODEL_FILE,
    epoch=10,
    audio_on=False,
    render_mode="human",
    use_lidar=False,
    score_limit=None,
):
    """Plays `epoch` episodes using a loaded checkpoint.

    Raises:
        FileNotFoundError: if no checkpoint exists at `model_file`.

    Returns:
        List of achieved scores, one entry per episode.
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"No checkpoint at '{model_file}'. Train an agent first and save it "
            f"with q_model.save_weights(...)."
        )

    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=audio_on,
        render_mode=render_mode,
        use_lidar=use_lidar,
        score_limit=score_limit,
    )

    q_model = build_model(env)
    q_model.load_weights(model_file)
    q_model.summary()

    scores = []
    for t in range(epoch):
        state, _ = env.reset()
        state = np.expand_dims(state, axis=0)
        info = {"score": 0}

        while True:
            action = q_model.get_action(state)
            action = np.array(action, copy=False, dtype=env.action_space.dtype)

            next_state, _, terminated, truncated, info = env.step(action)
            state = np.expand_dims(next_state, axis=0)

            if terminated or truncated:
                break

        scores.append(info["score"])
        print(f"Episode: {t}, Score: {info['score']}")

    env.close()
    print(f"Mean score over {epoch} episodes: {np.mean(scores):.2f}")
    return scores


def test_play():
    """Smoke test - skipped as long as no checkpoint exists."""
    import pytest

    if not os.path.exists(DEFAULT_MODEL_FILE):
        pytest.skip(f"No checkpoint at '{DEFAULT_MODEL_FILE}'")

    scores = play(
        epoch=1,
        audio_on=False,
        render_mode=None,
        use_lidar=False,
        score_limit=10,
    )
    assert len(scores) == 1


if __name__ == "__main__":
    play()
