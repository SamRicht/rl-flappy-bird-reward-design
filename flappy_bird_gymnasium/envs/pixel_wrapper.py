"""Observation wrapper that turns the rendered game screen into CNN input.

The environment itself only yields feature vectors (12 values) or LIDAR
readings (180 values). For a convolutional agent we instead take the frame
produced by `render_mode="rgb_array"` (288x512x3), convert it to grayscale and
shrink it to 84x84. Stacking 4 consecutive frames gives the network the motion
information (velocity) that a single image lacks.

Note: `FlappyBirdEnv.step()` calls `render()` only in "human" mode, so this
wrapper triggers the rendering itself on every observation.
"""

from typing import Tuple

import gymnasium
import numpy as np
from gymnasium.wrappers import FrameStackObservation
from PIL import Image

FRAME_SIZE = (84, 84)


class PixelObservation(gymnasium.ObservationWrapper):
    """Replaces the observation with a preprocessed grayscale frame.

    Args:
        env: A FlappyBird environment created with `render_mode="rgb_array"`.
        size (Tuple[int, int]): Output size as (width, height).
    """

    def __init__(self, env: gymnasium.Env, size: Tuple[int, int] = FRAME_SIZE):
        super().__init__(env)
        if env.render_mode != "rgb_array":
            raise ValueError(
                "PixelObservation requires render_mode='rgb_array', "
                f"got {env.render_mode!r}"
            )
        self._size = size
        # Shape is (height, width) as usual for image arrays.
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=(size[1], size[0]), dtype=np.uint8
        )

    def observation(self, observation):
        frame = self.env.render()  # (height, width, 3) uint8
        image = Image.fromarray(frame).convert("L").resize(self._size, Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def make_pixel_env(stack: int = 4, **kwargs) -> gymnasium.Env:
    """Creates a FlappyBird env whose observations are stacked 84x84 frames.

    The returned observation has shape `(stack, 84, 84)` and dtype uint8.

    `background="night"` is used on purpose: in grayscale the bird (~194) is
    nearly invisible on the flat fill (200) and on the "day" sky (~159), but
    stands out clearly on the dark night sky (~96). `use_lidar=False` keeps the
    standard reward function (the LIDAR variant adds a proximity penalty). Both
    can be overridden via `kwargs`.
    """
    kwargs.setdefault("background", "night")
    kwargs.setdefault("use_lidar", False)
    env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", **kwargs)
    env = PixelObservation(env)
    return FrameStackObservation(env, stack_size=stack)
