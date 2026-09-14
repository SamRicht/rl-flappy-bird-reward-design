#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""Implementation of a Flappy Bird OpenAI gymnasium environment that yields simple
numerical information about the game's state as observations.

Some of the code in this module is an adaption of the code in the `FlapPyBird`
GitHub repository by `sourahbhv` (https://github.com/sourabhv/FlapPyBird),
released under the MIT license.
"""

from enum import IntEnum
from itertools import cycle
from typing import Dict, Optional, Tuple, Union

import gymnasium
import numpy as np
import pygame

from flappy_bird_gymnasium.envs import utils
from flappy_bird_gymnasium.envs.constants import (
    BACKGROUND_WIDTH,
    BASE_WIDTH,
    FILL_BACKGROUND_COLOR,
    LIDAR_MAX_DISTANCE,
    PIPE_HEIGHT,
    PIPE_VEL_X,
    PIPE_WIDTH,
    PLAYER_ACC_Y,
    PLAYER_FLAP_ACC,
    PLAYER_HEIGHT,
    PLAYER_MAX_VEL_Y,
    PLAYER_PRIVATE_ZONE,
    PLAYER_ROT_THR,
    PLAYER_VEL_ROT,
    PLAYER_WIDTH,
)
from flappy_bird_gymnasium.envs.lidar import LIDAR
from flappy_bird_gymnasium.rl.rewards import RewardConfig, compute_reward


class Actions(IntEnum):
    """Possible actions for the player to take."""

    IDLE, FLAP = 0, 1


class FlappyBirdEnv(gymnasium.Env):
    """Flappy Bird Gymnasium environment that yields simple observations.

    The observations yielded by this environment are simple numerical
    information about the game's state. Specifically, the observations are:

        * Horizontal distance to the next pipe;
        * Difference between the player's y position and the next hole's y
          position.

    The reward received by the agent in each step is determined by a
    :class:`~flappy_bird_gymnasium.rl.rewards.RewardConfig`, which makes the
    reward function itself an object of study.  The default configuration
    reproduces the scheme used by the upstream project, so existing code and
    trained models behave exactly as before unless a config is passed in.

    Where the reward lives, for anyone changing it:

    * ``rl/rewards.py`` holds the weights, the named presets and the rule by
      which the terms combine.  That file is the one to edit.
    * ``step`` below only records *what happened* -- pipe passed, crashed,
      ceiling touched, flapped -- and hands those facts to ``compute_reward``.
      No weight appears in the game logic.
    * ``_shaping_term`` / ``_gap_potential`` add the optional dense signal used
      by the ``"shaped"`` preset; they are inert for every other scheme.

    Args:
        screen_size (Tuple[int, int]): The screen's width and height.
        normalize_obs (bool): If `True`, the observations will be normalized
            before being returned.
        pipe_gap (int): Space between a lower and an upper pipe.
        bird_color (str): Color of the flappy bird. The currently available
            colors are "yellow", "blue" and "red".
        pipe_color (str): Color of the pipes. The currently available colors are
            "green" and "red".
        background (Optional[str]): Type of background image. The currently
            available types are "day" and "night". If `None`, no background will
            be drawn.
        reward_config (Optional[RewardConfig]): The reward scheme to use. If
            `None`, the upstream scheme (alive = +0.1, pipe = +1.0, dead = -1.0,
            ceiling = -0.5) is used.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        screen_size: Tuple[int, int] = (288, 512),
        audio_on: bool = False,
        normalize_obs: bool = True,
        use_lidar: bool = True,
        pipe_gap: int = 100,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        render_mode: Optional[str] = None,
        background: Optional[str] = "day",
        score_limit: Optional[int] = None,
        debug: bool = False,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._debug = debug
        self._score_limit = score_limit
        # Defaults to the upstream reward scheme, which `RewardConfig()` mirrors.
        self._reward_config = reward_config or RewardConfig()
        self._prev_potential = 0.0
        self._flap_count = 0

        self.action_space = gymnasium.spaces.Discrete(2)
        if use_lidar:
            if normalize_obs:
                self.observation_space = gymnasium.spaces.Box(
                    0.0, 1.0, shape=(180,), dtype=np.float64
                )
            else:
                self.observation_space = gymnasium.spaces.Box(
                    0.0, np.inf, shape=(180,), dtype=np.float64
                )
        else:
            if normalize_obs:
                self.observation_space = gymnasium.spaces.Box(
                    -1.0, 1.0, shape=(12,), dtype=np.float64
                )
            else:
                self.observation_space = gymnasium.spaces.Box(
                    -np.inf, np.inf, shape=(12,), dtype=np.float64
                )

        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]
        self._normalize_obs = normalize_obs
        self._pipe_gap = pipe_gap
        self._audio_on = audio_on
        self._use_lidar = use_lidar
        self._sound_cache = None
        self._player_flapped = False
        self._player_idx_gen = cycle([0, 1, 2, 1])
        self._bird_color = bird_color
        self._pipe_color = pipe_color
        self._bg_type = background

        self._ground = {"x": 0, "y": self._screen_height * 0.79}
        self._base_shift = BASE_WIDTH - BACKGROUND_WIDTH

        if use_lidar:
            self._lidar = LIDAR(LIDAR_MAX_DISTANCE)
            self._get_observation = self._get_observation_lidar
        else:
            self._get_observation = self._get_observation_features

        if render_mode is not None:
            self._fps_clock = pygame.time.Clock()
            self._display = None
            self._surface = pygame.Surface(screen_size)
            self._images = utils.load_images(
                convert=False,
                bird_color=bird_color,
                pipe_color=pipe_color,
                bg_type=background,
            )
            if audio_on:
                self._sounds = utils.load_sounds()

    def step(
        self,
        action: Union[Actions, int],
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Given an action, updates the game state.

        Args:
            action (Union[FlappyBirdLogic.Actions, int]): The action taken by
                the agent. Zero (0) means "do nothing" and one (1) means "flap".

        Returns:
            A tuple containing, respectively:

                * an observation (horizontal distance to the next pipe
                  difference between the player's y position and the next hole's
                  y position)
                * a reward (alive = +0.1, pipe = +1.0, dead = -1.0)
                * a status report (`True` if the game is over and `False`
                  otherwise)
                * an info dictionary
        """
        """Given an action taken by the player, updates the game's state.

        Args:
            action (Union[FlappyBirdLogic.Actions, int]): The action taken by
                the player.

        Returns:
            `True` if the player is alive and `False` otherwise.
        """
        flapped = False
        passed_pipe = False

        self._sound_cache = None
        if action == Actions.FLAP:
            if self._player_y > -2 * PLAYER_HEIGHT:
                self._player_vel_y = PLAYER_FLAP_ACC
                self._player_flapped = True
                self._sound_cache = "wing"
                flapped = True
                self._flap_count += 1

        # check for score
        player_mid_pos = self._player_x + PLAYER_WIDTH / 2
        for pipe in self._upper_pipes:
            pipe_mid_pos = pipe["x"] + PIPE_WIDTH / 2
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                self._score += 1
                passed_pipe = True
                self._sound_cache = "point"

        # player_index base_x change
        if (self._loop_iter + 1) % 3 == 0:
            self._player_idx = next(self._player_idx_gen)

        self._loop_iter = (self._loop_iter + 1) % 30
        self._ground["x"] = -((-self._ground["x"] + 100) % self._base_shift)

        # rotate the player
        if self._player_rot > -90:
            self._player_rot -= PLAYER_VEL_ROT

        # player's movement
        if self._player_vel_y < PLAYER_MAX_VEL_Y and not self._player_flapped:
            self._player_vel_y += PLAYER_ACC_Y

        if self._player_flapped:
            self._player_flapped = False

            # more rotation to cover the threshold
            # (calculated in visible rotation)
            self._player_rot = 45

        self._player_y += min(
            self._player_vel_y, self._ground["y"] - self._player_y - PLAYER_HEIGHT
        )

        # move pipes to left
        for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
            up_pipe["x"] += PIPE_VEL_X
            low_pipe["x"] += PIPE_VEL_X

            # it is out of the screen
            if up_pipe["x"] < -PIPE_WIDTH:
                new_up_pipe, new_low_pipe = self._get_random_pipe()
                up_pipe["x"] = new_up_pipe["x"]
                up_pipe["y"] = new_up_pipe["y"]
                low_pipe["x"] = new_low_pipe["x"]
                low_pipe["y"] = new_low_pipe["y"]

        if self.render_mode == "human":
            self.render()

        obs, in_private_zone = self._get_observation()

        # check
        if self._debug and self._use_lidar:
            # sort pipes by the distance between pipe and agent
            up_pipe = sorted(
                self._upper_pipes,
                key=lambda x: np.sqrt(
                    (self._player_x - x["x"]) ** 2
                    + (self._player_y - (x["y"] + PIPE_HEIGHT)) ** 2
                ),
            )[0]
            # find ray closest to the obstacle
            min_index = np.argmin(obs)
            min_value = obs[min_index] * LIDAR_MAX_DISTANCE
            # mean approach to the obstacle
            if "pipe_mean_value" in self._statistics:
                self._statistics["pipe_mean_value"] = self._statistics[
                    "pipe_mean_value"
                ] * 0.99 + min_value * (1 - 0.99)
            else:
                self._statistics["pipe_mean_value"] = min_value

            # Nearest to the pipe
            if "pipe_min_value" in self._statistics:
                if min_value < self._statistics["pipe_min_value"]:
                    self._statistics["pipe_min_value"] = min_value
                    self._statistics["pipe_min_index"] = min_index
            else:
                self._statistics["pipe_min_value"] = min_value
                self._statistics["pipe_min_index"] = min_index

            # Nearest to the ground
            diff = np.abs(self._player_y - self._ground["y"])
            if "ground_min_value" in self._statistics:
                if diff < self._statistics["ground_min_value"]:
                    self._statistics["ground_min_value"] = diff
            else:
                self._statistics["ground_min_value"] = diff

        # agent touch the top of the screen as punishment
        touched_ceiling = self._player_y < 0

        # check for crash
        terminal = self._check_crash()
        if terminal:
            self._sound_cache = "hit"
            self._player_vel_y = 0
            if self._debug and self._use_lidar:
                if ((self._player_x + PLAYER_WIDTH) - up_pipe["x"]) > (0 + 5) and (
                    self._player_x - up_pipe["x"]
                ) < PIPE_WIDTH:
                    print("BETWEEN PIPES")
                elif ((self._player_x + PLAYER_WIDTH) - up_pipe["x"]) < (0 + 5):
                    print("IN FRONT OF")
                print(
                    f"obs: [{self._statistics['pipe_min_index']},"
                    f"{self._statistics['pipe_min_value']},"
                    f"{self._statistics['pipe_mean_value']}],"
                    f"Ground: {self._statistics['ground_min_value']}"
                )

        # The step only *observes* what happened and hands the facts over; the
        # weighting and the way the terms combine live in RewardConfig, so that
        # changing the reward never means touching the game logic. Swap schemes
        # with `gym.make("FlappyBird-v0", reward_config=RewardConfig.preset(...))`;
        # passing nothing reproduces the original rewards exactly.
        reward = compute_reward(
            self._reward_config,
            passed_pipe=passed_pipe,
            crashed=terminal,
            touched_ceiling=touched_ceiling,
            in_private_zone=in_private_zone,
            flapped=flapped,
            shaping=self._shaping_term(terminal),
        )

        info = {"score": self._score, "flaps": self._flap_count}

        return (
            obs,
            reward,
            terminal,
            (self._score_limit is not None) and (self._score >= self._score_limit),
            info,
        )

    def reset(self, seed=None, options=None):
        """Resets the environment (starts a new game)."""
        super().reset(seed=seed)

        # Player's info:
        self._player_x = int(self._screen_width * 0.2)
        self._player_y = int((self._screen_height - PLAYER_HEIGHT) / 2)
        self._player_vel_y = -9  # player"s velocity along Y
        self._player_rot = 45  # player"s rotation
        self._player_idx = 0
        self._loop_iter = 0
        self._score = 0
        self._flap_count = 0

        if self._debug and self._use_lidar:
            self._statistics = {}

        # Generate 3 new pipes to add to upper_pipes and lower_pipes lists
        new_pipe1 = self._get_random_pipe()
        new_pipe2 = self._get_random_pipe()
        new_pipe3 = self._get_random_pipe()

        # List of upper pipes:
        self._upper_pipes = [
            {"x": self._screen_width, "y": new_pipe1[0]["y"]},
            {
                "x": self._screen_width + (self._screen_width / 2),
                "y": new_pipe2[0]["y"],
            },
            {
                "x": self._screen_width + self._screen_width,
                "y": new_pipe3[0]["y"],
            },
        ]

        # List of lower pipes:
        self._lower_pipes = [
            {"x": self._screen_width, "y": new_pipe1[1]["y"]},
            {
                "x": self._screen_width + (self._screen_width / 2),
                "y": new_pipe2[1]["y"],
            },
            {
                "x": self._screen_width + self._screen_width,
                "y": new_pipe3[1]["y"],
            },
        ]

        if self.render_mode == "human":
            self.render()

        obs, _ = self._get_observation()
        self._prev_potential = (
            self._gap_potential() if self._reward_config.uses_shaping else 0.0
        )
        info = {"score": self._score, "flaps": self._flap_count}
        return obs, info

    def _shaping_term(self, terminal: bool) -> float:
        """Returns the potential difference ``gamma * Phi(s') - Phi(s)``.

        This is *reward shaping*: an extra reward that gives feedback on every
        single frame instead of only every ~36th one, when a pipe is passed.
        It compares how well the bird stood before the move with how well it
        stands after: flying towards the centre of the next gap scores a small
        positive term, drifting away from it a small negative one.

        The specific form is what makes it safe to add.  Summed over an episode
        the terms cancel out pairwise and leave only the constant ``-Phi(s_0)``
        (the "telescoping" property), and a constant cannot reorder policies by
        their return.  The shaping therefore speeds learning up *without
        changing which policy is optimal* -- the guarantee from Ng et al.
        (1999). 

        Two conditions are required for the guarantee, and both are easy to
        break by accident:

        1. The potential of a terminal state must be treated as zero, which is
           the ``0.0 if terminal`` below.  Using the real potential of the state
           the bird died in would leave a policy-dependent remainder.
        2. ``shaping_gamma`` must equal the discount factor the agent learns
           with.  Nothing here can check that, so the training entry point sets
           the two from the same value; see ``rl/train.py::config_from_args``.

        Args:
            terminal: Whether the step ended the episode.

        Returns:
            The shaping term, or ``0.0`` when shaping is disabled (the default,
            and the case for every reward preset except ``"shaped"``).
        """
        config = self._reward_config
        if not config.uses_shaping:
            return 0.0

        next_potential = 0.0 if terminal else self._gap_potential()
        shaping = config.shaping_gamma * next_potential - self._prev_potential
        # Remember Phi(s') so the next step can use it as its Phi(s). `reset`
        # primes this with Phi(s_0), otherwise the first step of an episode
        # would compare against a potential left over from the previous one.
        self._prev_potential = next_potential
        return shaping

    def _gap_potential(self) -> float:
        """Returns the potential ``Phi(s)``: how well the bird is currently placed.

        One number summarising the state, used only by the shaping term above.
        It is the vertical distance between the bird and the centre of the gap
        it is flying towards, negated and expressed in screen heights: ``0``
        exactly on the centre line, about ``-0.8`` at the far end of the screen.
        Negative throughout, so "less bad" means "better placed".

        The list comprehension selects *which* gap to aim at, and that is less
        obvious than it looks.  ``_upper_pipes`` holds three pipe pairs in no
        particular role order -- they are recycled to the right edge once they
        leave the screen, so any of the three may be the relevant one.  Keeping
        only pipes whose right edge is still at or ahead of the bird's right
        edge drops the ones already flown through; the nearest of the survivors
        is the target.  A fixed index would track the wrong pipe for part of
        every cycle.

        Returns:
            The potential, or ``0.0`` in the moment no pipe lies ahead.
        """
        player_right = self._player_x + PLAYER_WIDTH
        gaps = [
            # (x of the pair, vertical centre between the two pipe mouths)
            (up_pipe["x"], (up_pipe["y"] + PIPE_HEIGHT + low_pipe["y"]) / 2)
            for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes)
            # the pipe has not been passed yet
            if up_pipe["x"] + PIPE_WIDTH >= player_right
        ]
        if not gaps:
            return 0.0

        # Nearest pipe still ahead = the one the bird has to get through next.
        _, gap_center = min(gaps, key=lambda gap: gap[0])
        player_center = self._player_y + PLAYER_HEIGHT / 2
        return -abs(player_center - gap_center) / self._screen_height

    def render(self) -> None:
        """Renders the next frame."""
        if self.render_mode == "rgb_array":
            self._draw_surface(show_score=False, show_rays=False)
            # Flip the image to retrieve a correct aspect
            return np.transpose(pygame.surfarray.array3d(self._surface), axes=(1, 0, 2))
        else:
            self._draw_surface(show_score=True, show_rays=self._use_lidar)
            if self._display is None:
                self._make_display()

            self._update_display()
            self._fps_clock.tick(self.metadata["render_fps"])

    def close(self):
        """Closes the environment."""
        if self.render_mode is not None:
            pygame.display.quit()
            pygame.quit()
        super().close()

    def _get_random_pipe(self) -> Dict[str, int]:
        """Returns a randomly generated pipe."""
        # y of gap between upper and lower pipe
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = self.np_random.integers(0, len(gapYs))
        gap_y = gapYs[index]
        gap_y += int(self._ground["y"] * 0.2)

        pipe_x = self._screen_width + PIPE_WIDTH + (self._screen_width * 0.2)
        return [
            {"x": pipe_x, "y": gap_y - PIPE_HEIGHT},  # upper pipe
            {"x": pipe_x, "y": gap_y + self._pipe_gap},  # lower pipe
        ]

    def _check_crash(self) -> bool:
        """Returns True if player collides with the ground (base) or a pipe."""
        # if player crashes into ground
        if self._player_y + PLAYER_HEIGHT >= self._ground["y"] - 1:
            if self._debug and self._use_lidar:
                print("CRASH TO THE GROUND")
            return True
        else:
            player_rect = pygame.Rect(
                self._player_x, self._player_y, PLAYER_WIDTH, PLAYER_HEIGHT
            )

            for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
                # upper and lower pipe rects
                up_pipe_rect = pygame.Rect(
                    up_pipe["x"], up_pipe["y"], PIPE_WIDTH, PIPE_HEIGHT
                )
                low_pipe_rect = pygame.Rect(
                    low_pipe["x"], low_pipe["y"], PIPE_WIDTH, PIPE_HEIGHT
                )

                # check collision
                up_collide = player_rect.colliderect(up_pipe_rect)
                low_collide = player_rect.colliderect(low_pipe_rect)

                if self._debug and self._use_lidar:
                    if up_collide:
                        print("CRASH TO UPPER PIPE")
                        print(
                            f"up_pipe: {[up_pipe['x'], up_pipe['y']+PIPE_HEIGHT]},"
                            f"low_pipe: {low_pipe},"
                            f"player: [{self._player_x}, {self._player_y}]"
                        )
                        return True
                    if low_collide:
                        print("CRASH TO LOWER PIPE")
                        print(
                            f"up_pipe: {[up_pipe['x'], up_pipe['y']+PIPE_HEIGHT]},"
                            f"low_pipe: {low_pipe},"
                            f"player: [{self._player_x}, {self._player_y}]"
                        )
                        return True
                else:
                    if up_collide or low_collide:
                        return True

        return False

    def _get_observation_features(self) -> np.ndarray:
        pipes = []
        for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
            # the pipe is behind the screen?
            if low_pipe["x"] > self._screen_width:
                pipes.append((self._screen_width, 0, self._screen_height))
            else:
                pipes.append(
                    (low_pipe["x"], (up_pipe["y"] + PIPE_HEIGHT), low_pipe["y"])
                )

        pipes = sorted(pipes, key=lambda x: x[0])
        pos_y = self._player_y
        vel_y = self._player_vel_y
        rot = self._player_rot

        if self._normalize_obs:
            pipes = [
                (
                    h / self._screen_width,
                    v1 / self._screen_height,
                    v2 / self._screen_height,
                )
                for h, v1, v2 in pipes
            ]
            pos_y /= self._screen_height
            vel_y /= PLAYER_MAX_VEL_Y
            rot /= 90

        return (
            np.array(
                [
                    pipes[0][0],  # the last pipe's horizontal position
                    pipes[0][1],  # the last top pipe's vertical position
                    pipes[0][2],  # the last bottom pipe's vertical position
                    pipes[1][0],  # the next pipe's horizontal position
                    pipes[1][1],  # the next top pipe's vertical position
                    pipes[1][2],  # the next bottom pipe's vertical position
                    pipes[2][0],  # the next next pipe's horizontal position
                    pipes[2][1],  # the next next top pipe's vertical position
                    pipes[2][2],  # the next next bottom pipe's vertical position
                    pos_y,  # player's vertical position
                    vel_y,  # player's vertical velocity
                    rot,  # player's rotation
                ]
            ),
            False,  # no private zone without LIDAR readings
        )

    def _get_observation_lidar(self) -> np.ndarray:
        # obstacles
        distances = self._lidar.scan(
            self._player_x,
            self._player_y,
            self._player_rot,
            self._upper_pipes,
            self._lower_pipes,
            self._ground,
        )

        in_private_zone = bool(np.any(distances < PLAYER_PRIVATE_ZONE))

        if self._normalize_obs:
            distances = distances / LIDAR_MAX_DISTANCE

        return distances, in_private_zone

    def _make_display(self) -> None:
        """Initializes the pygame's display.

        Required for drawing images on the screen.
        """
        self._display = pygame.display.set_mode(
            (self._screen_width, self._screen_height)
        )
        for name, value in self._images.items():
            if value is None:
                continue

            if type(value) in (tuple, list):
                self._images[name] = tuple([img.convert_alpha() for img in value])
            else:
                self._images[name] = (
                    value.convert() if name == "background" else value.convert_alpha()
                )

    def _draw_score(self) -> None:
        """Draws the score in the center of the surface."""
        score_digits = [int(x) for x in list(str(self._score))]
        total_width = 0  # total width of all numbers to be printed

        for digit in score_digits:
            total_width += self._images["numbers"][digit].get_width()

        x_offset = (self._screen_width - total_width) / 2

        for digit in score_digits:
            self._surface.blit(
                self._images["numbers"][digit], (x_offset, self._screen_height * 0.1)
            )
            x_offset += self._images["numbers"][digit].get_width()

    def _draw_surface(self, show_score: bool = True, show_rays: bool = True) -> None:
        """Re-draws the renderer's surface.

        This method updates the renderer's surface by re-drawing it according to
        the current state of the game.

        Args:
            show_score (bool): Whether to draw the player's score or not.
        """
        # Background
        if self._images["background"] is not None:
            self._surface.blit(self._images["background"], (0, 0))
        else:
            self._surface.fill(FILL_BACKGROUND_COLOR)

        # Pipes
        for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
            self._surface.blit(self._images["pipe"][0], (up_pipe["x"], up_pipe["y"]))
            self._surface.blit(self._images["pipe"][1], (low_pipe["x"], low_pipe["y"]))

        # Base (ground)
        self._surface.blit(self._images["base"], (self._ground["x"], self._ground["y"]))

        # Getting player's rotation
        visible_rot = PLAYER_ROT_THR
        if self._player_rot <= PLAYER_ROT_THR:
            visible_rot = self._player_rot

        # LIDAR
        if show_rays:
            self._lidar.draw(self._surface, self._player_x, self._player_y)

            # Draw private zone
            target_rect = pygame.Rect(
                self._player_x - PLAYER_PRIVATE_ZONE,
                self._player_y - PLAYER_PRIVATE_ZONE,
                PLAYER_PRIVATE_ZONE * 2 + PLAYER_WIDTH,
                PLAYER_PRIVATE_ZONE * 2 + PLAYER_HEIGHT,
            )
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.draw.circle(
                shape_surf,
                "blue",
                (
                    PLAYER_PRIVATE_ZONE + PLAYER_WIDTH,
                    PLAYER_PRIVATE_ZONE + (PLAYER_HEIGHT / 2),
                ),
                PLAYER_PRIVATE_ZONE,
                1,
                draw_top_left=False,
                draw_top_right=True,
                draw_bottom_left=False,
                draw_bottom_right=True,
            )
            pygame.draw.circle(
                shape_surf,
                "blue",
                (PLAYER_PRIVATE_ZONE, PLAYER_PRIVATE_ZONE + (PLAYER_HEIGHT / 2)),
                PLAYER_PRIVATE_ZONE,
                1,
                draw_top_left=True,
                draw_top_right=False,
                draw_bottom_left=True,
                draw_bottom_right=False,
            )
            pygame.draw.circle(
                shape_surf,
                "blue",
                (PLAYER_PRIVATE_ZONE + (PLAYER_WIDTH / 2), PLAYER_PRIVATE_ZONE),
                PLAYER_PRIVATE_ZONE,
                1,
                draw_top_left=True,
                draw_top_right=True,
                draw_bottom_left=False,
                draw_bottom_right=False,
            )
            pygame.draw.circle(
                shape_surf,
                "blue",
                (
                    PLAYER_PRIVATE_ZONE + (PLAYER_WIDTH / 2),
                    PLAYER_PRIVATE_ZONE + PLAYER_HEIGHT,
                ),
                PLAYER_PRIVATE_ZONE,
                1,
                draw_top_left=False,
                draw_top_right=False,
                draw_bottom_left=True,
                draw_bottom_right=True,
            )
            rotated_surf = pygame.transform.rotate(shape_surf, visible_rot)
            self._surface.blit(
                rotated_surf, rotated_surf.get_rect(center=target_rect.center)
            )

        # Score
        # (must be drawn before the player, so the player overlaps it)
        if show_score:
            self._draw_score()

        # Player
        player_surface = pygame.transform.rotate(
            self._images["player"][self._player_idx],
            visible_rot,
        )
        player_surface_rect = player_surface.get_rect(
            topleft=(self._player_x, self._player_y)
        )
        self._surface.blit(player_surface, player_surface_rect)

    def _update_display(self) -> None:
        """Updates the display with the current surface of the renderer.

        A call to this method is usually preceded by a call to
        :meth:`.draw_surface()`. This method simply updates the display by
        showing the current state of the renderer's surface on it, it doesn't
        make any change to the surface.
        """
        if self._display is None:
            raise RuntimeError(
                "Tried to update the display, but a display hasn't been "
                "created yet! To create a display for the renderer, you must "
                "call the `make_display()` method."
            )

        pygame.event.get()
        self._display.blit(self._surface, [0, 0])
        pygame.display.update()

        # Sounds:
        if self._audio_on and self._sound_cache is not None:
            self._sounds[self._sound_cache].play()
