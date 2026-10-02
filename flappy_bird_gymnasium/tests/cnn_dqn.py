"""Convolutional Dueling-DQN for pixel observations (PyTorch).

Counterpart to `dueling.py`, which works on the feature/LIDAR vectors. This
network expects stacked grayscale frames of shape (batch, 4, H, W) as
produced by `flappy_bird_gymnasium.envs.pixel_wrapper.make_pixel_env`.

Architecture follows the "Nature DQN" (Mnih et al., 2015) with a dueling head
(Wang et al., 2016), i.e. the same Q = V + (A - mean(A)) scheme as in
`dueling.py`. The constructor options (`layer_norm`, `orthogonal_init`,
`fc_width`, `input_hw`) are ablations for the training study; all defaults
reproduce the original network exactly, so old checkpoints keep loading.
"""

import math

import torch
import torch.nn as nn

FRAME_STACK = 4


class DuelingCNN(nn.Module):
    """Dueling Q-network on the Nature-DQN convolution stack.

    Args:
        action_space: Number of discrete actions.
        in_channels: Stacked frames per observation.
        layer_norm: Insert `nn.LayerNorm` between the 512-unit linear layer
            and its ReLU. Damps the value divergence that shows up in the
            eval curves as a collapse after a run has reached its peak. The
            cost is negligible - the conv stack makes 83% of the FLOPs, the
            norm works on `fc_width` activations.
        orthogonal_init: Initialise every conv/linear layer of the trunk
            orthogonally with gain sqrt(2) and zero bias (PyTorch's default,
            Kaiming-uniform with a=sqrt(5), is poorly calibrated for ReLU
            nets). The heads V and A get gain 1.0 - no ReLU follows them.
        fc_width: Width of the linear layer after the conv stack. 95% of the
            parameters sit in this layer while it makes 17% of the compute;
            256 halves the parameter count. An ablation, not a speed knob.
        input_hw: (height, width) of one frame. The flatten size is computed
            from a dummy forward, so non-square observations work (no
            `nn.LazyLinear`: that would need a forward before `load_state_dict`
            and break `load_cnn`).
    """

    def __init__(
        self,
        action_space: int,
        in_channels: int = FRAME_STACK,
        layer_norm: bool = False,
        orthogonal_init: bool = False,
        fc_width: int = 512,
        input_hw=(84, 84),
    ):
        super().__init__()

        conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():  # 84x84 input -> 7x7 feature map -> 3136
            flat = conv(torch.zeros(1, in_channels, *input_hw)).shape[1]
        layers = [*conv, nn.Linear(flat, fc_width)]
        if layer_norm:
            layers.append(nn.LayerNorm(fc_width))
        layers.append(nn.ReLU())
        self.features = nn.Sequential(*layers)
        self.V = nn.Linear(fc_width, 1)
        self.A = nn.Linear(fc_width, action_space)

        if orthogonal_init:
            for layer in self.features:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                    nn.init.zeros_(layer.bias)
            for head in (self.V, self.A):
                nn.init.orthogonal_(head.weight, gain=1.0)
                nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frames are stored as uint8 (saves replay-buffer memory); scale to [0, 1].
        x = x.float() / 255.0
        x = self.features(x)
        V = self.V(x)
        A = self.A(x)
        return V + (A - A.mean(dim=-1, keepdim=True))

    @torch.no_grad()
    def get_action(self, state) -> int:
        """Greedy action for a single stacked observation of shape (4, H, W)."""
        device = next(self.parameters()).device
        state = torch.as_tensor(state, device=device).unsqueeze(0)
        return int(self(state).argmax(dim=-1).item())


def load_cnn(model_file: str, action_space: int = 2, input_hw=(84, 84)) -> DuelingCNN:
    """Loads a checkpoint into a `DuelingCNN` of the matching variant.

    The variant is read off the state dict, never guessed: a LayerNorm brings
    an extra 1-D `features.*.weight` of shape `(fc_width,)`, and `fc_width`
    is the output size of the linear layer that follows the flatten. A
    `config.json` next to the checkpoint would be the alternative, but the
    copy exported to `assets/model/` has none - hence the state dict.

    `input_hw` cannot be recovered from the checkpoint (only the flatten size
    is, and several frame sizes map to the same value), so the caller passes
    the frame size of the env the model plays on; a mismatch raises from
    `load_state_dict` with a shape error.

    Returns the network in eval mode.
    """
    state = torch.load(model_file, map_location="cpu")
    feature_weights = {
        k: v
        for k, v in state.items()
        if k.startswith("features.") and k.endswith("weight")
    }
    layer_norm = any(v.ndim == 1 for v in feature_weights.values())
    fc_width = next(v.shape[0] for v in feature_weights.values() if v.ndim == 2)
    model = DuelingCNN(
        action_space,
        in_channels=state["features.0.weight"].shape[1],
        layer_norm=layer_norm,
        fc_width=int(fc_width),
        input_hw=tuple(input_hw),
    )
    model.load_state_dict(state)
    model.eval()
    return model
