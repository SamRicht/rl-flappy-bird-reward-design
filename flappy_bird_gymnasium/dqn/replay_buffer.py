"""Uniform experience replay."""

from typing import Optional, Tuple

import numpy as np


class ReplayBuffer:
    """A fixed-size ring buffer of transitions, sampled uniformly.

    Everything is kept in pre-allocated numpy arrays, so memory usage is
    constant and known up-front: for ``capacity`` transitions of a 12-dim
    observation that is roughly ``capacity * 110`` bytes.
    """

    def __init__(self, capacity: int, obs_dim: int, seed: Optional[int] = None):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        # stored as float so it can be used directly in the Bellman target
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Stores one transition, overwriting the oldest one when full.

        Args:
            done: whether the episode *terminated* (the bird crashed). A
                truncation (score limit reached) must be passed as `False`,
                otherwise the agent learns that success looks like dying.
        """
        i = self._pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draws a batch of transitions with replacement."""
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )
