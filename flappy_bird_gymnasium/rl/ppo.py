"""A self-contained PPO implementation (Schulman et al., 2017).

The algorithm follows the widely used "PPO with clipped surrogate objective and
generalised advantage estimation" recipe:

1. Roll out the current policy for ``n_steps`` transitions in each of ``n_envs``
   parallel environments.
2. Estimate advantages with GAE(``gamma``, ``gae_lambda``).
3. Perform ``n_epochs`` passes of minibatch SGD on the clipped surrogate
   objective, a (optionally clipped) value loss and an entropy bonus.

Everything is deliberately written out rather than delegated to a library, so
that each hyperparameter of the study has a visible place in the code.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    """Hyperparameters of the PPO agent.

    Attributes:
        total_steps: Total number of environment steps to train for, summed over
            all parallel environments.
        n_envs: Number of environments stepped in parallel.
        n_steps: Rollout length per environment; the batch size per update is
            ``n_envs * n_steps``.
        n_epochs: Passes of minibatch SGD over each collected batch.
        n_minibatches: Number of minibatches each batch is split into.
        lr: Learning rate of the Adam optimiser.
        anneal_lr: Linearly decay the learning rate to zero over training.
        gamma: Discount factor.
        gae_lambda: Bias/variance trade-off of generalised advantage estimation;
            ``1.0`` yields Monte-Carlo returns, ``0.0`` a one-step TD estimate.
        clip_eps: Clipping range of the surrogate objective.
        clip_vloss: Apply the same clipping to the value loss.
        ent_coef: Weight of the entropy bonus; larger values keep the policy
            stochastic for longer and delay premature convergence.
        vf_coef: Weight of the value loss.
        max_grad_norm: Threshold for global gradient-norm clipping.
        target_kl: If set, stop the epochs of an update early once the
            approximate KL divergence between old and new policy exceeds it.
        norm_adv: Normalise advantages per minibatch to zero mean, unit variance.
        hidden_sizes: Widths of the hidden layers of the MLP.
        shared_backbone: Share the hidden layers between actor and critic instead
            of using two separate networks.
        seed: Seed for network initialisation, action sampling and environments.
    """

    total_steps: int = 1_000_000
    n_envs: int = 8
    n_steps: int = 256
    n_epochs: int = 4
    n_minibatches: int = 4
    lr: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    norm_adv: bool = True
    hidden_sizes: Tuple[int, ...] = (64, 64)
    shared_backbone: bool = False
    seed: int = 0

    @property
    def batch_size(self) -> int:
        return self.n_envs * self.n_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.n_minibatches

    @property
    def n_updates(self) -> int:
        return max(1, self.total_steps // self.batch_size)

    def __post_init__(self) -> None:
        if self.batch_size % self.n_minibatches != 0:
            raise ValueError(
                f"batch size {self.batch_size} is not divisible by "
                f"n_minibatches {self.n_minibatches}"
            )
        # Tuples survive a JSON round-trip as lists; normalise back.
        self.hidden_sizes = tuple(self.hidden_sizes)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PPOConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    """Orthogonal initialisation, the default for PPO's MLP policies.

    Orthogonal weights with a gain of ``sqrt(2)`` preserve the scale of the
    activations through ``tanh`` layers; the policy head uses a much smaller gain
    so that the initial policy is close to uniform.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(sizes: List[int], activation: type = nn.Tanh) -> List[nn.Module]:
    layers: List[nn.Module] = []
    for in_size, out_size in zip(sizes[:-1], sizes[1:]):
        layers += [layer_init(nn.Linear(in_size, out_size)), activation()]
    return layers


class ActorCritic(nn.Module):
    """MLP actor-critic for a discrete action space.

    Args:
        obs_dim: Dimensionality of the observation vector.
        n_actions: Size of the discrete action space.
        hidden_sizes: Widths of the hidden layers.
        shared_backbone: If `True`, actor and critic share the hidden layers and
            only the output heads are separate.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        shared_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.shared_backbone = shared_backbone
        sizes = [obs_dim, *hidden_sizes]

        if shared_backbone:
            self.backbone = nn.Sequential(*_mlp(sizes))
            self.actor_body = nn.Identity()
            self.critic_body = nn.Identity()
        else:
            self.backbone = nn.Identity()
            self.actor_body = nn.Sequential(*_mlp(sizes))
            self.critic_body = nn.Sequential(*_mlp(sizes))

        # A near-zero gain keeps the initial policy close to uniform, which
        # avoids a large policy change in the very first update.
        self.actor_head = layer_init(nn.Linear(hidden_sizes[-1], n_actions), std=0.01)
        self.critic_head = layer_init(nn.Linear(hidden_sizes[-1], 1), std=1.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns ``V(s)`` for a batch of observations."""
        return self.critic_head(self.critic_body(self.backbone(obs))).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples an action (or evaluates a given one) and returns its value.

        Args:
            obs: Batch of observations.
            action: If given, these actions are evaluated instead of sampling
                new ones -- used during the update phase.

        Returns:
            Tuple of ``(action, log_prob, entropy, value)``.
        """
        features = self.backbone(obs)
        logits = self.actor_head(self.actor_body(features))
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        value = self.critic_head(self.critic_body(features)).squeeze(-1)
        return action, dist.log_prob(action), dist.entropy(), value


@dataclass
class RolloutBuffer:
    """Fixed-size storage for one PPO batch, laid out as ``(n_steps, n_envs)``."""

    n_steps: int
    n_envs: int
    obs_dim: int
    device: torch.device

    obs: torch.Tensor = field(init=False)
    actions: torch.Tensor = field(init=False)
    log_probs: torch.Tensor = field(init=False)
    rewards: torch.Tensor = field(init=False)
    values: torch.Tensor = field(init=False)
    # `dones` marks the *start* of an episode boundary, i.e. whether the state at
    # this index is the first of a fresh episode.
    dones: torch.Tensor = field(init=False)
    # Bootstrap value of the successor state, non-zero only where an episode was
    # truncated rather than terminated.
    truncation_values: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.n_steps, self.n_envs)
        self.obs = torch.zeros((*shape, self.obs_dim), device=self.device)
        self.actions = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros(shape, device=self.device)
        self.rewards = torch.zeros(shape, device=self.device)
        self.values = torch.zeros(shape, device=self.device)
        self.dones = torch.zeros(shape, device=self.device)
        self.truncation_values = torch.zeros(shape, device=self.device)

    def compute_advantages(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes GAE advantages and value targets.

        Args:
            last_value: ``V(s_T)`` for the state following the last stored step.
            last_done: Whether that state starts a fresh episode.
            gamma: Discount factor.
            gae_lambda: GAE decay.

        Returns:
            Tuple of ``(advantages, returns)``, both shaped ``(n_steps, n_envs)``.
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.n_envs, device=self.device)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # A truncated episode has a successor state that simply was not
            # observed; its value must still be bootstrapped, otherwise hitting
            # the step limit would look exactly like dying.
            rewards = self.rewards[t] + gamma * self.truncation_values[t]

            delta = rewards + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        return advantages, advantages + self.values

    def flatten(self) -> Dict[str, torch.Tensor]:
        """Collapses the ``(n_steps, n_envs)`` layout into a flat batch."""
        return {
            "obs": self.obs.reshape(-1, self.obs_dim),
            "actions": self.actions.reshape(-1),
            "log_probs": self.log_probs.reshape(-1),
            "values": self.values.reshape(-1),
        }


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Runs the PPO update for one collected batch.

    Args:
        model: The actor-critic being optimised.
        optimizer: Its optimiser.
        batch: Flattened rollout tensors from :meth:`RolloutBuffer.flatten`.
        advantages: Flattened GAE advantages.
        returns: Flattened value targets.
        config: Hyperparameters.
        rng: Source of the minibatch permutation.

    Returns:
        Diagnostics of the update (losses, approximate KL, clip fraction,
        explained variance of the value function).
    """
    batch_size = config.batch_size
    indices = np.arange(batch_size)

    clip_fractions: List[float] = []
    approx_kls: List[float] = []
    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []
    epochs_run = 0
    stop_early = False

    for _ in range(config.n_epochs):
        rng.shuffle(indices)
        epochs_run += 1

        for start in range(0, batch_size, config.minibatch_size):
            mb = indices[start : start + config.minibatch_size]

            _, new_log_prob, entropy, new_value = model.get_action_and_value(
                batch["obs"][mb], batch["actions"][mb]
            )
            log_ratio = new_log_prob - batch["log_probs"][mb]
            ratio = log_ratio.exp()

            with torch.no_grad():
                # Schulman's low-variance estimator of KL(old || new); unlike
                # `-log_ratio.mean()` it is non-negative and far less noisy.
                approx_kls.append(((ratio - 1) - log_ratio).mean().item())
                clip_fractions.append(
                    ((ratio - 1.0).abs() > config.clip_eps).float().mean().item()
                )

            mb_advantages = advantages[mb]
            if config.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Clipped surrogate objective: the pessimistic (maximum) of the two
            # losses bounds how far the policy may move per update.
            pg_loss_unclipped = -mb_advantages * ratio
            pg_loss_clipped = -mb_advantages * torch.clamp(
                ratio, 1 - config.clip_eps, 1 + config.clip_eps
            )
            policy_loss = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()

            if config.clip_vloss:
                v_loss_unclipped = (new_value - returns[mb]) ** 2
                v_clipped = batch["values"][mb] + torch.clamp(
                    new_value - batch["values"][mb], -config.clip_eps, config.clip_eps
                )
                v_loss_clipped = (v_clipped - returns[mb]) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * ((new_value - returns[mb]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = (
                policy_loss
                - config.ent_coef * entropy_loss
                + config.vf_coef * value_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy_loss.item())

        if config.target_kl is not None and approx_kls[-1] > config.target_kl:
            stop_early = True
            break

    y_pred = batch["values"].cpu().numpy()
    y_true = returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),
        "clip_fraction": float(np.mean(clip_fractions)),
        "explained_variance": float(explained_var),
        "epochs_run": float(epochs_run),
        "stopped_early": float(stop_early),
    }
