"""The tabular Q-learning agent."""

import json
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Sequence, Tuple, Union

import numpy as np

from flappy_bird_gymnasium.qlearning.config import QLearningConfig
from flappy_bird_gymnasium.qlearning.discretize import Discretizer, build_discretizer


class QLearningAgent:
    """Q-table over the rasterised state space, updated in place.

    The table is a dense array, not a dict. The state space is small and
    bounded anyway, a dense array makes the update a couple of indexing
    operations, and it lets us count how many states were never visited. A
    dict can't answer that, because the unseen keys simply don't exist.

    Attributes:
        q: ``(n_states, n_actions)`` action values.
        counts: ``(n_states, n_actions)`` visit counts. Used for the coverage
            statistic, and useful on their own when judging a rasterisation.
    """

    def __init__(
        self,
        discretizer: Discretizer,
        n_actions: int,
        config: Optional[QLearningConfig] = None,
    ):
        self.cfg = config or QLearningConfig()
        self.discretizer = discretizer
        self.n_actions = int(n_actions)
        self.q = np.full(
            (discretizer.n_states, self.n_actions), self.cfg.q_init, dtype=np.float64
        )
        self.counts = np.zeros((discretizer.n_states, self.n_actions), dtype=np.int64)
        self.train_steps = 0
        # own RNG, so the agent doesn't depend on whatever else draws from the
        # global NumPy stream
        self._rng = np.random.default_rng(self.cfg.seed)
        self.extra: dict = {}
        # transitions waiting for their n-step return. With n_step=1 this holds
        # at most one entry and the update is plain Q-learning.
        self._pending: Deque[Tuple[int, int, float]] = deque()
        # one-entry memo for the state index, see _index_of
        self._cached_obs = None
        self._cached_index = -1

    # ------------------------------------------------------------------
    # schedules
    # ------------------------------------------------------------------
    def epsilon(self, step: int) -> float:
        """Exploration rate at ``step``: linear decay, then held."""
        frac = min(1.0, step / max(1, self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + frac * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    def learning_rate(self, step: int) -> float:
        """Step size at ``step``: linear decay, then held.

        Decaying lets the table settle once the policy stops changing. With a
        constant step size the values keep bouncing, because the environment
        is stochastic and every visit pulls an entry towards a different
        sample.

        ``learning_rate_decay_steps = None`` means the decay spans the whole
        run, so it follows the budget instead of ending after a fixed fraction.
        """
        horizon = self.cfg.learning_rate_decay_steps or self.cfg.total_steps
        frac = min(1.0, step / max(1, horizon))
        return self.cfg.learning_rate + frac * (
            self.cfg.learning_rate_end - self.cfg.learning_rate
        )

    # ------------------------------------------------------------------
    # acting
    # ------------------------------------------------------------------
    def act(self, obs: Sequence[float], epsilon: float = 0.0) -> int:
        """Epsilon-greedy action for a raw observation.

        Takes the raw observation and rasterises internally, so that
        ``lambda obs: agent.act(obs)`` works as a policy for
        :func:`flappy_bird_gymnasium.rl.rollout.rollout` and the agent gets
        measured by the same code as the other methods.

        Ties are broken at random. That matters more than it looks: with
        ``q_init = 0.0`` all actions of an unvisited state are tied, and a
        plain argmax would always return action 0, i.e. never flap. The bird
        would drop to the ground in every such state and the table would
        never see the situations it needs to learn.
        """
        if epsilon > 0.0 and self._rng.random() < epsilon:
            return int(self._rng.integers(self.n_actions))

        row = self.q[self._index_of(obs)]
        best = np.flatnonzero(row == row.max())
        if best.size == 1:
            return int(best[0])
        return int(best[self._rng.integers(best.size)])

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def update(
        self,
        obs: Sequence[float],
        action: int,
        reward: float,
        next_obs: Sequence[float],
        terminated: bool,
        step: int,
    ) -> float:
        """Applies one update and returns the TD error.

        Args:
            terminated: Whether the episode ended because the bird died. Do
                not pass ``True`` for a truncation by the frame limit: the
                episode stopped, but the future wasn't worthless, and a zero
                continuation value there would teach the agent that surviving
                long is bad.

        Returns:
            The TD error of the applied update, or 0.0 while the n-step
            buffer is still filling.
        """
        self.train_steps = step
        self._pending.append((self._index_of(obs), int(action), float(reward)))
        next_state = self._index_of(next_obs)

        if terminated:
            # nothing to bootstrap from after a death, and every buffered
            # transition leads into it within its own horizon, so the whole
            # buffer drains here
            td_error = 0.0
            while self._pending:
                td_error = self._apply_oldest(next_state, False, step)
            return td_error

        if len(self._pending) >= self.cfg.n_step:
            return self._apply_oldest(next_state, True, step)
        return 0.0

    def finish_truncated_episode(self, last_obs: Sequence[float], step: int) -> None:
        """Drains the n-step buffer at a truncation, keeping the bootstrap.

        Hitting the frame limit stops the episode but doesn't make the future
        worthless. Just dropping the buffer would throw away the last
        ``n_step - 1`` transitions of every truncated episode, and late in a
        run nearly every episode is truncated.
        """
        last_state = self._index_of(last_obs)
        while self._pending:
            self._apply_oldest(last_state, True, step)

    def _apply_oldest(self, next_state: int, bootstrap: bool, step: int) -> float:
        """Backs the buffered rewards up onto the oldest pending transition."""
        state, action, _ = self._pending[0]

        horizon = len(self._pending)
        discounted_return = 0.0
        for index, (_, _, reward) in enumerate(self._pending):
            discounted_return += (self.cfg.gamma**index) * reward
        if bootstrap:
            discounted_return += (self.cfg.gamma**horizon) * self._continuation(
                next_state, step
            )

        td_error = discounted_return - float(self.q[state, action])
        self.q[state, action] += self._step_size(state, action, step) * td_error
        self.counts[state, action] += 1
        self._pending.popleft()
        return td_error

    def _continuation(self, next_state: int, step: int) -> float:
        """Value of continuing from ``next_state`` under the update rule."""
        row = self.q[next_state]
        if self.cfg.algo == "expected_sarsa":
            # expectation under the epsilon-greedy policy: uniform with
            # probability epsilon, greedy otherwise. Every greedy action has
            # the max value, so how the tie was broken doesn't matter here.
            epsilon = self.epsilon(step)
            return epsilon * float(row.mean()) + (1.0 - epsilon) * float(row.max())
        return float(row.max())

    def _index_of(self, obs: Sequence[float]) -> int:
        """State index of ``obs``, reusing the last result for the same array.

        A training step asks for the same observation three times (once in
        ``act``, twice in ``update`` as current and then as next state), and
        rasterising is about a third of the loop's runtime. The environment
        returns a fresh array each step and never mutates one in place, so
        comparing by identity is enough. A miss just recomputes.
        """
        if obs is self._cached_obs:
            return self._cached_index
        index = self.discretizer.index(obs)
        self._cached_obs = obs
        self._cached_index = index
        return index

    def _step_size(self, state: int, action: int, step: int) -> float:
        """Step size for one entry.

        In ``count`` mode each entry anneals on its own visit count instead
        of on training progress. That matters at the end of a long run: a
        global schedule forces an entry seen twenty times to settle at the
        same moment as one seen twenty thousand times.
        """
        if self.cfg.learning_rate_mode == "count":
            visits = float(self.counts[state, action])
            alpha = self.cfg.learning_rate / (1.0 + visits) ** (
                self.cfg.learning_rate_exponent
            )
            return max(alpha, self.cfg.learning_rate_end)
        return self.learning_rate(step)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    @property
    def coverage(self) -> float:
        """Fraction of state-action pairs that were updated at least once.

        Not comparable across rasterisations: the reachable states form a
        thin manifold in (dx, dy, vel), so a finer grid adds far more cells
        than reachable cells and coverage drops no matter how well the run
        explored.
        """
        return float(np.count_nonzero(self.counts) / self.counts.size)

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path], **extra) -> None:
        """Writes the table plus everything needed to rebuild the agent.

        The config travels with the checkpoint so that ``evaluate.py`` can
        rebuild the same environment the agent was trained in. Measuring a
        policy under a different reward scheme or pipe gap than it was
        trained with gives a meaningless number.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            q=self.q,
            counts=self.counts,
            config=json.dumps(self.cfg.to_dict()),
            extra=json.dumps(extra, default=str),
            n_actions=self.n_actions,
            train_steps=self.train_steps,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "QLearningAgent":
        """Rebuilds an agent from a checkpoint written by :meth:`save`."""
        # allow_pickle stays off, a checkpoint is data and not code
        data = np.load(Path(path), allow_pickle=False)
        config = QLearningConfig.from_dict(json.loads(str(data["config"].item())))
        agent = cls(build_discretizer(config), int(data["n_actions"]), config)

        expected = (agent.discretizer.n_states, agent.n_actions)
        if data["q"].shape != expected:
            raise ValueError(
                f"checkpoint table has shape {data['q'].shape}, but the stored "
                f"configuration describes {expected} -- the file is corrupt or "
                f"was written by an incompatible version"
            )

        agent.q = data["q"].astype(np.float64)
        agent.counts = data["counts"].astype(np.int64)
        agent.train_steps = int(data["train_steps"])
        agent.extra = json.loads(str(data["extra"].item()))
        return agent
