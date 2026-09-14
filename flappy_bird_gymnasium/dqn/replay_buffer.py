"""Uniform experience replay, with n-step return accumulation."""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np


class ReplayBuffer:
    """A fixed-size ring buffer of transitions, sampled uniformly.

    Everything is kept in pre-allocated numpy arrays, so memory usage is
    constant and known up-front: for ``capacity`` transitions of a 12-dim
    observation that is roughly ``capacity * 115`` bytes.

    Each transition carries its own ``discount`` (``gamma ** horizon``) rather
    than relying on a single global gamma. That is what lets n-step and
    1-step transitions coexist in the same buffer -- necessary because the
    transitions flushed at the end of an episode have a shorter horizon than
    the rest.
    """

    def __init__(self, capacity: int, obs_dim: int, seed: Optional[int] = None):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        # stored as float so they can be used directly in the Bellman target
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.discounts = np.zeros(self.capacity, dtype=np.float32)

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
        discount: float,
    ) -> None:
        """Stores one transition, overwriting the oldest one when full.

        Args:
            reward: the (already discounted) n-step return of the transition.
            next_obs: the state reached after the n steps.
            done: whether the episode *terminated* (the bird crashed). A
                truncation (step or score limit reached) must be passed as
                `False`, otherwise the agent learns that success looks like
                dying.
            discount: ``gamma ** horizon``, applied to the bootstrap value.
        """
        i = self._pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.discounts[i] = discount

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Draws a batch of transitions with replacement."""
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
            self.discounts[idx],
        )


class NStepAccumulator:
    """Folds single-step transitions into n-step transitions.

    With ``n = 1`` this is a pass-through. For larger ``n`` the stored reward
    becomes ``r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1}`` and the
    bootstrap starts from the state ``n`` steps later, discounted by
    ``gamma ** n``.

    Why it matters here: in Flappy Bird the bird needs about 50 frames to reach
    the first pipe, so the ``+1`` for passing it has to travel ~50 Bellman
    backups before it can influence the first action of the episode. n-step
    returns move that credit ``n`` frames at a time instead of one.
    """

    def __init__(self, n_step: int, gamma: float):
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")
        self.n_step = n_step
        self.gamma = gamma
        self._queue: deque = deque()

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> List[Tuple]:
        """Adds a step and returns whichever n-step transitions are complete.

        On termination the whole queue is flushed: every state still waiting
        gets the (shorter) return that actually followed it, which is exact
        because there is no future left to bootstrap from.
        """
        self._queue.append((obs, action, reward, next_obs, terminated))

        if terminated:
            return self._drain()
        if len(self._queue) >= self.n_step:
            ready = self._build(self.n_step)
            self._queue.popleft()
            return [ready]
        return []

    def flush(self) -> List[Tuple]:
        """Drains the queue at a *truncation* (an episode cut short, not ended).

        The emitted transitions keep ``terminated=False``, so their bootstrap
        survives -- the future was not worthless, the episode merely stopped.
        """
        return self._drain()

    def reset(self) -> None:
        """Drops anything still queued, e.g. when starting a new episode."""
        self._queue.clear()

    def _drain(self) -> List[Tuple]:
        out = []
        while self._queue:
            out.append(self._build(len(self._queue)))
            self._queue.popleft()
        return out

    def _build(self, horizon: int) -> Tuple:
        obs, action = self._queue[0][0], self._queue[0][1]
        n_step_return = 0.0
        for k in range(horizon):
            n_step_return += (self.gamma**k) * self._queue[k][2]
        _, _, _, last_next_obs, last_terminated = self._queue[horizon - 1]
        return (
            obs,
            action,
            n_step_return,
            last_next_obs,
            last_terminated,
            self.gamma**horizon,
        )
