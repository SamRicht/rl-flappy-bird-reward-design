"""Configurable reward functions for the Flappy Bird environment.

The original environment hard-codes a single reward function.  For the study of
*reward design* we need to swap it out, so every reward term is collected in a
:class:`RewardConfig` and the environment merely evaluates that configuration.

Two composition modes exist:

``"override"``
    The scheme used by the upstream project: the terms form a priority chain and
    only the highest-priority term that fired contributes to the reward
    (crash > ceiling > pipe > private zone > alive).  Passing a pipe in the same
    step in which the bird dies therefore yields ``-1``, not ``0``.

``"additive"``
    The more conventional formulation: every term that fired is summed up.
    Passing a pipe while dying yields ``pipe + death``.

The distinction matters: under ``"override"`` the agent can never collect the
pipe bonus on its final step, which slightly discourages squeezing through a gap
at the last moment.

Additionally a *potential-based shaping* term (Ng et al., 1999) can be enabled.
Because it has the form ``F(s, s') = gamma * Phi(s') - Phi(s)``, it provably
leaves the optimal policy unchanged while densifying the learning signal -- a
useful control condition when arguing that a shaped reward did not simply buy
performance by changing the task.
"""

from dataclasses import asdict, dataclass, replace
from typing import Dict


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the individual reward terms.

    Attributes:
        name: Identifier used for logging and result directories.
        mode: ``"override"`` (priority chain) or ``"additive"`` (sum).
        alive: Granted on every step in which nothing else happened.
        pipe: Granted when the bird passes a pipe.
        death: Granted when the bird crashes.
        ceiling: Granted while the bird is above the top of the screen.
        private_zone: Granted when a LIDAR ray falls below the safety distance.
            Only has an effect when the environment runs with ``use_lidar=True``.
        shaping_coef: Scale of the potential-based shaping term.  ``0.0``
            disables shaping.
        shaping_gamma: Discount factor used inside the shaping term.  For the
            policy-invariance guarantee to hold this must equal the discount
            factor of the learning algorithm.
        flap_cost: Added on every step in which the agent flaps.  Negative
            values make flapping "expensive" and bias the policy towards gliding.
    """

    name: str = "legacy"
    mode: str = "override"
    alive: float = 0.1
    pipe: float = 1.0
    death: float = -1.0
    ceiling: float = -0.5
    private_zone: float = -0.5
    shaping_coef: float = 0.0
    shaping_gamma: float = 0.99
    flap_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("override", "additive"):
            raise ValueError(
                f"mode must be 'override' or 'additive', got {self.mode!r}"
            )

    @property
    def uses_shaping(self) -> bool:
        return self.shaping_coef != 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "RewardConfig":
        return cls(**data)  # type: ignore[arg-type]

    @classmethod
    def preset(cls, name: str, **overrides: object) -> "RewardConfig":
        """Returns a named preset, optionally with individual terms replaced."""
        if name not in PRESETS:
            raise KeyError(
                f"unknown reward preset {name!r}; available: {sorted(PRESETS)}"
            )
        config = PRESETS[name]
        return replace(config, **overrides) if overrides else config


#: The reward schemes compared in the study.
PRESETS: Dict[str, RewardConfig] = {
    # The upstream reward, reproduced bit for bit.  Baseline of the comparison.
    "legacy": RewardConfig(name="legacy"),
    # Same terms, but summed instead of overridden.  Isolates the effect of the
    # composition mode alone.
    "additive": RewardConfig(name="additive", mode="additive"),
    # The "pure" task description: a point per pipe, a penalty for dying, and
    # nothing else.  Hardest to learn, but the only reward that expresses the
    # objective without bias.
    "sparse": RewardConfig(
        name="sparse",
        mode="additive",
        alive=0.0,
        pipe=1.0,
        death=-1.0,
        ceiling=0.0,
        private_zone=0.0,
    ),
    # Survival only -- no pipe bonus at all.  Tests whether staying alive is a
    # sufficient proxy objective, since in Flappy Bird surviving *requires*
    # passing pipes.
    "survival": RewardConfig(
        name="survival",
        mode="additive",
        alive=0.1,
        pipe=0.0,
        death=-1.0,
        ceiling=0.0,
        private_zone=0.0,
    ),
    # Sparse reward plus potential-based shaping towards the centre of the gap.
    # Policy-invariant by construction, so any improvement is an optimisation
    # effect rather than a change of the task.
    "shaped": RewardConfig(
        name="shaped",
        mode="additive",
        alive=0.0,
        pipe=1.0,
        death=-1.0,
        ceiling=0.0,
        private_zone=0.0,
        shaping_coef=1.0,
    ),
    # Strongly asymmetric penalty for dying.  Expected to produce a risk-averse
    # policy that hugs the centre of the gap.
    "risk_averse": RewardConfig(
        name="risk_averse",
        mode="additive",
        alive=0.1,
        pipe=1.0,
        death=-5.0,
        ceiling=-0.5,
        private_zone=0.0,
    ),
    # Flapping costs energy.  Expected to reduce the flap rate and produce a
    # visibly "smoother" trajectory at comparable score.
    "energy": RewardConfig(
        name="energy",
        mode="additive",
        alive=0.1,
        pipe=1.0,
        death=-1.0,
        ceiling=-0.5,
        private_zone=0.0,
        flap_cost=-0.02,
    ),
}


def compute_reward(
    config: RewardConfig,
    *,
    passed_pipe: bool,
    crashed: bool,
    touched_ceiling: bool,
    in_private_zone: bool,
    flapped: bool,
    shaping: float,
) -> float:
    """Evaluates ``config`` for a single environment step.

    Args:
        config: The reward scheme to evaluate.
        passed_pipe: Whether the bird passed a pipe in this step.
        crashed: Whether the bird collided with a pipe or the ground.
        touched_ceiling: Whether the bird is above the top of the screen.
        in_private_zone: Whether an obstacle is inside the LIDAR safety zone.
        flapped: Whether the agent's action was executed as a flap.
        shaping: The pre-computed potential difference
            ``gamma * Phi(s') - Phi(s)``, already zero when shaping is disabled.

    Returns:
        The scalar reward of the step.
    """
    if config.mode == "override":
        # Priority chain, highest priority last (it wins):
        # crash > ceiling > pipe > private zone > alive.
        if passed_pipe:
            reward = config.pipe
        elif in_private_zone:
            reward = config.private_zone
        else:
            reward = config.alive
        if touched_ceiling:
            reward = config.ceiling
        if crashed:
            reward = config.death
    else:
        reward = 0.0
        if not crashed:
            reward += config.alive
        if passed_pipe:
            reward += config.pipe
        if crashed:
            reward += config.death
        if touched_ceiling:
            reward += config.ceiling
        if in_private_zone:
            reward += config.private_zone

    if flapped:
        reward += config.flap_cost

    return reward + config.shaping_coef * shaping
