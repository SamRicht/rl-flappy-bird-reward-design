"""Tests the pixel observation wrapper and the convolutional Q-network, and
lets a trained CNN agent play (`flappy_bird_gymnasium --mode cnn`).
"""

import json
import os

import gymnasium
import numpy as np
import pytest

import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.pixel_wrapper import make_pixel_env
from flappy_bird_gymnasium.envs.utils import MODEL_PATH

DEFAULT_MODEL_FILE = os.path.join(MODEL_PATH, "cnn_dqn.pt")


def play(
    model_file=DEFAULT_MODEL_FILE,
    epoch=10,
    render=True,
    score_limit=None,
    **env_kwargs,
):
    """Plays `epoch` episodes greedily with a trained CNN checkpoint.

    The pixel wrapper needs `render_mode="rgb_array"`, so the env cannot open
    its own window. With `render=True` we open one here and blit the frame.
    `env_kwargs` are passed to `make_pixel_env` (e.g. `crop_ground=False` for
    checkpoints trained before the ground crop was introduced).

    Raises:
        FileNotFoundError: if no checkpoint exists at `model_file`.

    Returns:
        List of achieved scores, one entry per episode.
    """
    import pygame

    from flappy_bird_gymnasium.tests.cnn_dqn import load_cnn

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"No checkpoint at '{model_file}'. Train an agent first with "
            f"`python -m flappy_bird_gymnasium.train_cnn`."
        )

    env = make_pixel_env(score_limit=score_limit, **env_kwargs)
    model = load_cnn(
        model_file, int(env.action_space.n), input_hw=env.observation_space.shape[1:]
    )

    display, clock = None, None
    if render:
        pygame.init()
        clock = pygame.time.Clock()

    scores = []
    for t in range(epoch):
        obs, _ = env.reset()
        info = {"score": 0}
        while True:
            action = model.get_action(obs)
            obs, _, terminated, truncated, info = env.step(action)

            if render:
                frame = env.render()  # (height, width, 3)
                if display is None:
                    display = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
                pygame.event.get()
                pygame.surfarray.blit_array(display, np.transpose(frame, (1, 0, 2)))
                pygame.display.update()
                clock.tick(env.metadata["render_fps"])

            if terminated or truncated:
                break

        scores.append(info["score"])
        print(f"Episode: {t}, Score: {info['score']}")

    env.close()
    if render:
        pygame.quit()
    print(f"Mean score over {epoch} episodes: {np.mean(scores):.2f}")
    return scores


@pytest.mark.parametrize("size", [(84, 84), (84, 128)])  # (width, height)
def test_pixel_env_observation_shape(size):
    env = make_pixel_env(size=size)
    shape = (4, size[1], size[0])
    obs, _ = env.reset(seed=0)
    assert obs.shape == shape
    assert obs.dtype == np.uint8
    assert env.observation_space.shape == shape

    obs, _, _, _, _ = env.step(0)
    assert obs.shape == shape
    # The frame must contain something (bird, pipes, ground), not a flat image.
    assert obs[-1].min() != obs[-1].max()
    env.close()

    torch = pytest.importorskip("torch")
    from flappy_bird_gymnasium.tests.cnn_dqn import DuelingCNN

    model = DuelingCNN(2, input_hw=shape[1:])
    assert model(torch.as_tensor(obs).unsqueeze(0)).shape == (1, 2)


def _frames_and_stacks(stride, steps, seed=0):
    """Single frames from PixelObservation and the stacks make_pixel_env
    builds from the same episode (same seed, same actions)."""
    from flappy_bird_gymnasium.envs.pixel_wrapper import PixelObservation

    rng = np.random.default_rng(seed)
    actions = [int(rng.random() < 0.1) for _ in range(steps)]
    single = PixelObservation(
        gymnasium.make(
            "FlappyBird-v0",
            render_mode="rgb_array",
            background="night",
            use_lidar=False,
        )
    )
    stacked = make_pixel_env(stride=stride)
    frames, stacks = [], []
    f, _ = single.reset(seed=seed)
    o, _ = stacked.reset(seed=seed)
    frames.append(f)
    stacks.append(o)
    for a in actions:
        f, _, term, trunc, _ = single.step(a)
        o, _, term2, trunc2, _ = stacked.step(a)
        assert (term, trunc) == (term2, trunc2)
        frames.append(f)
        stacks.append(o)
        if term or trunc:
            break
    single.close()
    stacked.close()
    return frames, stacks


@pytest.mark.parametrize("stride", [1, 3])
def test_frame_stack_picks_strided_frames(stride):
    """The stack at step t holds frames t, t-k, t-2k, t-3k, padded with frame 0."""
    frames, stacks = _frames_and_stacks(stride, steps=40)
    assert len(stacks) > 3 * stride + 1, "episode too short for the test"
    for t, stack in enumerate(stacks):
        for j, k in enumerate(reversed(range(4))):
            assert np.array_equal(stack[j], frames[max(0, t - k * stride)])


def test_strided_frame_stack_equals_gymnasium_at_stride_1():
    """StridedFrameStack must reproduce FrameStackObservation exactly."""
    from flappy_bird_gymnasium.envs.pixel_wrapper import (
        PixelObservation,
        StridedFrameStack,
    )

    ours = StridedFrameStack(
        PixelObservation(
            gymnasium.make(
                "FlappyBird-v0",
                render_mode="rgb_array",
                background="night",
                use_lidar=False,
            )
        ),
        stack=4,
        stride=1,
    )
    ref = make_pixel_env()  # FrameStackObservation
    assert ours.observation_space == ref.observation_space
    a, _ = ours.reset(seed=3)
    b, _ = ref.reset(seed=3)
    assert np.array_equal(a, b)
    for action in [0, 0, 1, 0, 0, 0, 1, 0]:
        a, *_ = ours.step(action)
        b, *_ = ref.step(action)
        assert np.array_equal(a, b)
    ours.close()
    ref.close()


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


def test_cnn_default_is_unchanged():
    """Guard for the study: without flags the network is exactly the old one."""
    torch = pytest.importorskip("torch")
    from flappy_bird_gymnasium.tests.cnn_dqn import DuelingCNN

    model = DuelingCNN(2)
    assert sum(p.numel() for p in model.parameters()) == 1_685_667
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in model.features)
    # State-dict layout of the original network (conv x3, linear at index 7).
    assert list(model.state_dict())[:2] == ["features.0.weight", "features.0.bias"]
    assert model.state_dict()["features.7.weight"].shape == (512, 3136)


@pytest.mark.parametrize("layer_norm", [False, True])
def test_load_cnn_roundtrip(tmp_path, layer_norm):
    torch = pytest.importorskip("torch")
    from flappy_bird_gymnasium.tests.cnn_dqn import DuelingCNN, load_cnn

    model = DuelingCNN(2, layer_norm=layer_norm)
    model.eval()
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    loaded = load_cnn(str(path))
    assert not loaded.training
    has_norm = any(isinstance(m, torch.nn.LayerNorm) for m in loaded.features)
    assert has_norm == layer_norm
    x = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
    assert torch.equal(model(x), loaded(x))


def test_orthogonal_init():
    torch = pytest.importorskip("torch")
    from flappy_bird_gymnasium.tests.cnn_dqn import DuelingCNN

    model = DuelingCNN(2, orthogonal_init=True)
    W = model.features[7].weight  # (512, 3136): rows orthogonal, gain sqrt(2)
    assert torch.allclose(W @ W.T, 2 * torch.eye(512), atol=1e-4)
    for name, param in model.named_parameters():
        if name.endswith("bias"):
            assert torch.count_nonzero(param) == 0, name
    # Heads use gain 1: A is (2, 512) -> A A^T = I.
    A = model.A.weight
    assert torch.allclose(A @ A.T, torch.eye(2), atol=1e-5)


def test_fc_width():
    torch = pytest.importorskip("torch")
    from flappy_bird_gymnasium.tests.cnn_dqn import DuelingCNN

    width = 256
    model = DuelingCNN(2, fc_width=width)
    out = model(torch.zeros(1, 4, 84, 84, dtype=torch.uint8))
    assert out.shape == (1, 2)
    conv = 4 * 32 * 8 * 8 + 32 + 32 * 64 * 4 * 4 + 64 + 64 * 64 * 3 * 3 + 64
    expected = conv + (3136 * width + width) + (width + 1) + (2 * width + 2)
    assert sum(p.numel() for p in model.parameters()) == expected


@pytest.mark.parametrize("stride", [1, 3])
def test_frame_replay_buffer_reconstructs_env_stacks(stride):
    """The buffer must hand back exactly the stacks the env produced.

    Runs longer than the capacity so the ring wraps, and with random actions
    the bird dies often, so episode starts (padding) and episode boundaries
    are covered as well. With a stride the buffer has to pick the same
    frames as the env wrapper - a mismatch would not crash, it would
    silently train the network on stacks it never acts on.
    """
    pytest.importorskip("torch")
    from flappy_bird_gymnasium.train_cnn import FrameReplayBuffer

    capacity, steps = 200, 350
    env = make_pixel_env(stride=stride)
    buffer = FrameReplayBuffer(
        capacity, env.observation_space.shape, "cpu", stride=stride
    )

    rng = np.random.default_rng(0)
    history = []  # (obs, action, reward, next_obs, done) as seen by the env
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        action = int(rng.random() < 0.1)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, done)
        history.append((obs.copy(), action, reward, next_obs.copy(), done))
        obs = next_obs
        if done:
            obs, _ = env.reset()
    env.close()

    assert any(h[4] for h in history), "no episode ended, test is too weak"
    lo, hi = buffer._valid_range()
    assert lo == steps - capacity + 3 * stride and hi == steps - 1

    times = rng.integers(lo, hi, size=100)
    obs_b, action_b, ret_b, next_b, done_b, horizon_b = buffer._build(times)
    for k, t in enumerate(times):
        obs, action, reward, next_obs, done = history[t]
        assert np.array_equal(obs_b[k], obs)
        assert action_b[k] == action
        assert ret_b[k] == np.float32(reward)
        assert done_b[k] == float(done)
        assert horizon_b[k] == 1
        if not done:
            assert np.array_equal(next_b[k], next_obs)

    # The reset padding is the interesting case; make sure it was exercised.
    starts = [t for t in range(lo, hi) if buffer.pos[t % capacity] < 3 * stride]
    assert starts
    obs_b, *_ = buffer._build(starts)
    for k, t in enumerate(starts):
        assert np.array_equal(obs_b[k], history[t][0])


def test_frame_replay_buffer_n_step():
    """n-step return, horizon and done against a brute-force loop."""
    pytest.importorskip("torch")
    from flappy_bird_gymnasium.train_cnn import FrameReplayBuffer

    gamma, n_step, capacity, steps = 0.9, 3, 64, 100
    rng = np.random.default_rng(1)
    rewards = rng.normal(size=steps).astype(np.float32)
    dones = rng.random(steps) < 0.15
    buffer = FrameReplayBuffer(capacity, (4, 2, 2), "cpu", gamma, n_step)
    for t in range(steps):
        frame = np.full((4, 2, 2), t % 256, dtype=np.uint8)
        buffer.add(frame, t % 2, rewards[t], dones[t])

    lo, hi = buffer._valid_range()
    assert hi == steps - n_step
    times = np.arange(lo, hi)
    _, _, ret, next_obs, done, horizon = buffer._build(times)

    assert (horizon < n_step).any(), "no window crosses an episode end"
    for k, t in enumerate(times):
        expected, discount = 0.0, 1.0
        for j in range(n_step):
            expected += discount * rewards[t + j]
            discount *= gamma
            if dones[t + j]:
                break
        assert horizon[k] == j + 1
        assert done[k] == float(dones[t + j])
        assert ret[k] == pytest.approx(expected, rel=1e-5)
        # The bootstrap state is the stack at t + horizon (frame value = time).
        assert next_obs[k][-1, 0, 0] == (t + j + 1) % 256


def test_frame_replay_buffer_bootstraps_truncation():
    """A truncated step keeps its bootstrap, from the episode's final frame.

    Treating the frame limit as a crash would teach the network that flying
    well for long ends badly; the extra slot holding the final frame must
    never be sampled as a transition of its own.
    """
    pytest.importorskip("torch")
    from flappy_bird_gymnasium.train_cnn import FrameReplayBuffer

    def frame(value):
        return np.full((4, 2, 2), value, dtype=np.uint8)

    buffer = FrameReplayBuffer(32, (4, 2, 2), "cpu", gamma=0.9, n_step=3)
    # episode 1: frames 0, 1, 2, truncated on the step from 2 -> final frame 9
    for t in range(3):
        truncated = t == 2
        buffer.add(frame(t), 0, 1.0, False, truncated, frame(9) if truncated else None)
    # episode 2: frames 20, 21, ..., long enough to sample from
    for t in range(6):
        buffer.add(frame(20 + t), 1, 0.5, False)

    assert buffer.t == 10 and not buffer.valid[3]
    assert buffer.pos[3] == 3, "the final frame belongs to the truncated episode"
    assert buffer.pos[4] == 0, "the next episode starts after it"

    _, _, ret, next_obs, done, horizon = buffer._build([1, 2])
    # from t=1: r1 + 0.9 r2, cut at the truncation, bootstrapped from frame 9
    assert horizon.tolist() == [2, 1]
    assert ret[0] == pytest.approx(1.0 + 0.9)
    assert done.tolist() == [0.0, 0.0]
    assert next_obs[0][-1, 0, 0] == 9 and next_obs[1][-1, 0, 0] == 9

    np.random.seed(0)
    for _ in range(20):
        obs, *_ = buffer.sample(16)
        assert 9 not in obs[:, -1, 0, 0].tolist()

    with pytest.raises(ValueError):
        buffer.add(frame(0), 0, 0.0, False, True)


def test_quick_eval_uses_the_run_config():
    """quick_eval() must see the frames the network was trained on, and
    write exactly the shared eval.csv columns."""
    pytest.importorskip("torch")
    from flappy_bird_gymnasium.evaluate_cnn import CnnConfig
    from flappy_bird_gymnasium.rl.runlog import EVAL_LOG_FIELDS
    from flappy_bird_gymnasium.tests.cnn_dqn import DuelingCNN
    from flappy_bird_gymnasium.train_cnn import quick_eval

    config = CnnConfig(frame_stride=3, obs_size=(128, 84), max_episode_steps=200)
    actor = DuelingCNN(2, input_hw=(128, 84))
    actor.eval()
    stats = quick_eval(actor, config, episodes=2)
    assert ["step", *stats] == EVAL_LOG_FIELDS
    assert stats["mean_length"] <= 200
    assert stats == quick_eval(actor, config, episodes=2), "paired seeds"


def test_cnn_env_is_the_shared_one():
    """make_cnn_env: shared frame limit and reward, pixel observations."""
    from flappy_bird_gymnasium.evaluate_cnn import CnnConfig, make_cnn_env

    config = CnnConfig(reward_preset="shaped", max_episode_steps=5, obs_size=(128, 84))
    env = make_cnn_env(config)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 128, 84)
    assert env.unwrapped._reward_config.name == "shaped"
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated
    # record.py draws the GIF from render()
    assert env.render().shape == (512, 288, 3)
    env.close()


def _fake_run(run_dir, torch, old_layout=False):
    from flappy_bird_gymnasium.evaluate_cnn import CnnConfig
    from flappy_bird_gymnasium.tests.cnn_dqn import DuelingCNN

    run_dir.mkdir(parents=True)
    torch.save(DuelingCNN(2).state_dict(), run_dir / "best.pt")
    torch.save(DuelingCNN(2).state_dict(), run_dir / "final.pt")
    config = CnnConfig(reward_preset="shaped", seed=1)
    if old_layout:
        data = {"args": {"reward": "shaped", "seed": 1, "gamma": 0.99}}
        (run_dir / "log.csv").write_text(
            "step,episode,score,return,length,flaps,epsilon,loss\n"
            + "".join(f"{50 * (i + 1)},{i},{i},0.0,50,5,0.1,0.01\n" for i in range(30))
        )
        (run_dir / "eval.csv").write_text("step,mean,median,min,max\n100,3.5,3,1,6\n")
    else:
        data = {"config": config.to_dict()}
    (run_dir / "config.json").write_text(json.dumps(data))


def test_run_loader(tmp_path):
    torch = pytest.importorskip("torch")
    from flappy_bird_gymnasium.rl.analysis.runs import open_env, resolve_loader

    loader = resolve_loader("cnn")
    _fake_run(tmp_path / "shaped_seed1", torch)
    loaded = loader.load(tmp_path / "shaped_seed1" / loader.default_checkpoint)
    assert loaded.config.reward_preset == "shaped" and loaded.config.seed == 1
    env = open_env(loaded, max_episode_steps=10)
    obs, _ = env.reset(seed=0)
    assert loaded.policy(obs) in (0, 1)
    env.close()


def test_backfill_old_run(tmp_path):
    torch = pytest.importorskip("torch")
    from flappy_bird_gymnasium.evaluate_cnn import CnnConfig, backfill_run
    from flappy_bird_gymnasium.rl.analysis.runs import collect_runs, read_column

    run = tmp_path / "shaped_seed1"
    _fake_run(run, torch, old_layout=True)
    assert collect_runs(tmp_path) == {}
    assert backfill_run(run)
    assert collect_runs(tmp_path) == {"shaped": [run]}
    assert backfill_run(run) is None, "a second backfill does nothing"

    steps, flap = read_column(run, "flap_rate")
    assert len(steps) == 30 and flap[0] == pytest.approx(0.1)
    assert read_column(run, "median_score", "eval.csv")[1].tolist() == [3.0]
    assert (run / "eval_old.csv").exists()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["backfilled"] and summary["steps_to_1"] is not None
    config = CnnConfig.from_run(run)
    assert config.reward_preset == "shaped" and config.obs_size == (84, 84)


def test_play_cnn():
    """Smoke test - skipped as long as no checkpoint exists."""
    pytest.importorskip("torch")
    if not os.path.exists(DEFAULT_MODEL_FILE):
        pytest.skip(f"No checkpoint at '{DEFAULT_MODEL_FILE}'")

    scores = play(epoch=1, render=False, score_limit=10)
    assert len(scores) == 1
