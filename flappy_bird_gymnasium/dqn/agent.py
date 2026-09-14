"""The DQN agent: acting, storing transitions and the learning step."""

import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flappy_bird_gymnasium.dqn.config import DQNConfig
from flappy_bird_gymnasium.dqn.model import build_q_network
from flappy_bird_gymnasium.dqn.replay_buffer import NStepAccumulator, ReplayBuffer


class DQNAgent:
    """Deep Q-Network with a target network, replay and optional Double DQN."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: Optional[DQNConfig] = None,
        device: Optional[str] = None,
    ):
        self.cfg = config or DQNConfig()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.q_net = build_q_network(
            obs_dim, n_actions, self.cfg.hidden, self.cfg.dueling
        ).to(self.device)

        # The target network provides the bootstrap value. Freezing it for a
        # while is what keeps the regression target from chasing its own tail.
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.cfg.learning_rate
        )
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim, seed=self.cfg.seed)
        self._n_step = NStepAccumulator(self.cfg.n_step, self.cfg.gamma)
        self._rng = np.random.default_rng(self.cfg.seed)
        self.train_steps = 0

    # ------------------------------------------------------------------ acting

    def epsilon(self, step: int) -> float:
        """Linear decay from `epsilon_start` to `epsilon_end`."""
        frac = min(1.0, step / max(1, self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + frac * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Picks an action epsilon-greedily (greedy for `epsilon=0`)."""
        if epsilon > 0.0 and self._rng.random() < epsilon:
            return int(self._rng.integers(self.n_actions))

        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        q_values = self.q_net(state.unsqueeze(0))
        return int(q_values.argmax(dim=-1).item())

    def remember(self, obs, action, reward, next_obs, terminated) -> None:
        """Feeds a step to the n-step accumulator and stores what it completes.

        Pass `terminated` only -- a `truncated` episode must be ended with
        `finish_truncated_episode`, so that its last transitions keep their
        bootstrap instead of being marked as deaths.
        """
        for transition in self._n_step.push(obs, action, reward, next_obs, terminated):
            self.buffer.add(*transition)

    def finish_truncated_episode(self) -> None:
        """Flushes the accumulator when an episode was cut short, not ended."""
        for transition in self._n_step.flush():
            self.buffer.add(*transition)

    # ---------------------------------------------------------------- learning

    def learn(self) -> Optional[Dict[str, float]]:
        """One gradient step on a replay batch.

        Returns `None` while the buffer is still filling up.
        """
        if len(self.buffer) < max(self.cfg.learning_starts, self.cfg.batch_size):
            return None

        batch = self.buffer.sample(self.cfg.batch_size)
        obs, actions, rewards, next_obs, dones, discounts = (
            torch.as_tensor(item, device=self.device) for item in batch
        )

        # Q(s, a) for the actions that were actually taken
        q_taken = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                # online net selects the action, target net evaluates it --
                # this is what removes the max-operator's overestimation bias
                next_actions = self.q_net(next_obs).argmax(dim=-1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(dim=-1).values
            # `discounts` is gamma ** horizon, stored per transition, because
            # the n-step horizon is shorter for the steps flushed at the end
            # of an episode.
            targets = rewards + discounts * (1.0 - dones) * next_q

        if self.cfg.huber_loss:
            loss = F.smooth_l1_loss(q_taken, targets)
        else:
            loss = F.mse_loss(q_taken, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_interval == 0:
            self.sync_target()

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_taken.mean().item()),
            "target_mean": float(targets.mean().item()),
        }

    def sync_target(self) -> None:
        """Copies the online weights into the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------ persistence

    def save(self, path, **extra) -> None:
        """Writes weights, optimizer state and config to `path`."""
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg.to_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "train_steps": self.train_steps,
                **extra,
            },
            path,
        )

    @classmethod
    def load(cls, path, device: Optional[str] = None) -> "DQNAgent":
        """Recreates an agent from a checkpoint written by `save`."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = DQNConfig.from_dict(checkpoint["config"])
        agent = cls(
            checkpoint["obs_dim"], checkpoint["n_actions"], config, device=device
        )
        agent.q_net.load_state_dict(checkpoint["q_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.train_steps = checkpoint.get("train_steps", 0)
        return agent
