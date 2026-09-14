"""Q-networks for the feature-based observation space."""

from typing import Sequence

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int = None) -> nn.Sequential:
    layers = []
    last = in_dim
    for units in hidden:
        layers += [nn.Linear(last, units), nn.ReLU()]
        last = units
    if out_dim is not None:
        layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Plain MLP mapping an observation to one Q-value per action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DuelingQNetwork(nn.Module):
    """Splits the head into a state value V(s) and an advantage A(s, a).

    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a). In Flappy Bird most states are
    ones where the action barely matters, and separating "how good is this
    position" from "which action is better here" learns noticeably faster.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.body = _mlp(obs_dim, hidden)
        self.value = nn.Linear(hidden[-1], 1)
        self.advantage = nn.Linear(hidden[-1], n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.body(obs)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


def build_q_network(
    obs_dim: int,
    n_actions: int,
    hidden: Sequence[int] = (256, 256),
    dueling: bool = True,
) -> nn.Module:
    """Creates the dueling or the plain Q-network."""
    if dueling:
        return DuelingQNetwork(obs_dim, n_actions, hidden)
    return QNetwork(obs_dim, n_actions, hidden)
