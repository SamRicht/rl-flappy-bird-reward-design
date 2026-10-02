"""Observation wrapper that turns the rendered game screen into CNN input.

The environment itself only yields feature vectors (12 values) or LIDAR
readings (180 values). For a convolutional agent we instead take the frame
drawn for `render_mode="rgb_array"` (288x512x3), convert it to grayscale and
shrink it to 84x84. Stacking 4 consecutive frames gives the network the motion
information (velocity) that a single image lacks.

Note: `FlappyBirdEnv.step()` calls `render()` only in "human" mode, so this
wrapper triggers the drawing itself on every observation.
"""

from collections import deque
from typing import Callable, Tuple

import gymnasium
import numpy as np
import pygame
from gymnasium.wrappers import FrameStackObservation

FRAME_SIZE = (84, 84)

# ITU-R 601 luma weights, same as PIL's "L" conversion.
_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class PixelObservation(gymnasium.ObservationWrapper):
    """Replaces the observation with a preprocessed grayscale frame.

    Args:
        env: A FlappyBird environment created with `render_mode="rgb_array"`.
        size (Tuple[int, int]): Output size as (width, height).
        crop_ground (bool): Cut off the ground strip (bottom ~21% of the
            screen) before scaling. Nothing happens down there except the
            crash, and the screen is squeezed ~6x vertically but only ~3.4x
            horizontally, so every row of vertical resolution counts in
            Flappy Bird.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        size: Tuple[int, int] = FRAME_SIZE,
        crop_ground: bool = True,
    ):
        super().__init__(env)
        if env.render_mode != "rgb_array":
            raise ValueError(
                "PixelObservation requires render_mode='rgb_array', "
                f"got {env.render_mode!r}"
            )
        self._size = size
        self._crop_ground = crop_ground
        # Shape is (height, width) as usual for image arrays.
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=(size[1], size[0]), dtype=np.uint8
        )

    def observation(self, observation):
        # Draw directly onto the env's pygame surface and scale it there.
        # This is ~5x faster than `env.render()` + PIL, because it avoids
        # copying the full 288x512 frame into numpy before downscaling.
        env = self.env.unwrapped
        env._draw_surface(show_score=False, show_rays=False)
        surface = env._surface
        if self._crop_ground:
            width, _ = surface.get_size()
            surface = surface.subsurface((0, 0, width, int(env._ground["y"])))
        small = pygame.transform.smoothscale(surface, self._size)
        rgb = pygame.surfarray.pixels3d(small)  # (width, height, 3), no copy
        gray = rgb.astype(np.float32).dot(_GRAY_WEIGHTS).astype(np.uint8)
        return np.ascontiguousarray(gray.T)  # -> (height, width)


class StridedFrameStack(gymnasium.Wrapper):
    """Stacks frames `t, t-k, t-2k, ...` for a temporal stride `k`.

    Four consecutive frames span only 3 steps. The bird accelerates by ~0.2
    observation pixels per step^2, so the curvature of its path - what tells
    "still rising" from "about to tip over" - is ~0.2 px inside such a stack
    and drowns in the resampling noise of `smoothscale`. A stride of `k`
    widens the window to `(stack - 1) * k` steps without changing the decision
    frequency (an *action repeat*, the Atari convention, would widen it too but
    also halve the number of decisions per pipe; not implemented here).

    Padding at a reset matches `FrameStackObservation(padding_type="reset")`:
    the first frame of the episode stands in for every frame before it. The
    replay buffer in `train_cnn.FrameReplayBuffer` rebuilds these stacks from
    single frames and must use the same stride and padding.

    Args:
        env: Env whose observations are single frames of shape (H, W).
        stack (int): Frames per observation.
        stride (int): Steps between the stacked frames.
    """

    def __init__(self, env: gymnasium.Env, stack: int = 4, stride: int = 1):
        super().__init__(env)
        if stack < 1 or stride < 1:
            raise ValueError(f"stack and stride must be >= 1, got {stack}, {stride}")
        self._stack = stack
        self._stride = stride
        # Newest frame last; holds exactly the window the stack reaches back.
        self._frames = deque(maxlen=(stack - 1) * stride + 1)
        low = np.repeat(env.observation_space.low[None], stack, axis=0)
        high = np.repeat(env.observation_space.high[None], stack, axis=0)
        self.observation_space = gymnasium.spaces.Box(
            low, high, dtype=env.observation_space.dtype
        )

    def _observation(self):
        picks = [-1 - k * self._stride for k in reversed(range(self._stack))]
        return np.stack([self._frames[i] for i in picks])

    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        for _ in range(self._frames.maxlen):
            self._frames.append(frame)
        return self._observation(), info

    def step(self, action):
        frame, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(frame)
        return self._observation(), reward, terminated, truncated, info


def make_pixel_env(
    stack: int = 4,
    stride: int = 1,
    crop_ground: bool = True,
    size: Tuple[int, int] = FRAME_SIZE,
    **kwargs,
) -> gymnasium.Env:
    """Creates a FlappyBird env whose observations are stacked grayscale frames.

    The returned observation has shape `(stack, height, width)` and dtype
    uint8, by default `(4, 84, 84)`.

    `background="night"` is used on purpose: in grayscale the bird (~194) is
    nearly invisible on the flat fill (200) and on the "day" sky (~159), but
    stands out clearly on the dark night sky (~96). `use_lidar=False` keeps the
    standard reward function (the LIDAR variant adds a proximity penalty). Both
    can be overridden via `kwargs`.

    Args:
        stack: Frames per observation.
        stride: Steps between stacked frames, see `StridedFrameStack`. With
            the default 1 the original `FrameStackObservation` is used, so
            existing runs are reproduced bit for bit.
        crop_ground: See `PixelObservation`. Checkpoints trained before the
            ground crop was introduced need `crop_ground=False`.
        size: Frame size as (width, height). The screen is squeezed 4.8x
            vertically but only 3.4x horizontally at 84x84, and the vertical
            axis carries all the relevant information; (84, 128) (width,
            height) encodes a 1-px bird movement ~50% more strongly for ~60%
            more FLOPs.

    A checkpoint is only valid for the `stride` and `size` it was trained
    with: both change what the network sees, and neither is stored in the
    checkpoint (`config.json` of the run has them).
    """
    kwargs.setdefault("background", "night")
    kwargs.setdefault("use_lidar", False)
    env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", **kwargs)
    return pixel_wrapper(stack, stride, crop_ground, size)(env)


def pixel_wrapper(
    stack: int = 4,
    stride: int = 1,
    crop_ground: bool = True,
    size: Tuple[int, int] = FRAME_SIZE,
) -> Callable[[gymnasium.Env], gymnasium.Env]:
    """The observation wrappers of `make_pixel_env`, as one callable.

    For `flappy_bird_gymnasium.rl.rollout.make_env(..., wrapper=...)`, which
    builds the env with the shared reward scheme and frame limit and only
    needs to know how the CNN sees it. The env it is applied to must have
    `render_mode="rgb_array"`.
    """

    def wrap(env: gymnasium.Env) -> gymnasium.Env:
        env = PixelObservation(env, size=size, crop_ground=crop_ground)
        if stride == 1:
            return FrameStackObservation(env, stack_size=stack)
        return StridedFrameStack(env, stack=stack, stride=stride)

    return wrap
