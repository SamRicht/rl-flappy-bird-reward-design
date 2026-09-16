"""Convolutional Dueling-DQN for pixel observations (PyTorch).

Counterpart to `dueling.py`, which works on the feature/LIDAR vectors. This
network expects stacked grayscale frames of shape (batch, 4, 84, 84) as
produced by `flappy_bird_gymnasium.envs.pixel_wrapper.make_pixel_env`.

Architecture follows the "Nature DQN" (Mnih et al., 2015) with a dueling head
(Wang et al., 2016), i.e. the same Q = V + (A - mean(A)) scheme as in
`dueling.py`.
"""

import torch
import torch.nn as nn

FRAME_STACK = 4


class DuelingCNN(nn.Module):
    def __init__(self, action_space: int, in_channels: int = FRAME_STACK):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),  # 84x84 input -> 7x7 feature map
            nn.ReLU(),
        )
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, action_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frames are stored as uint8 (saves replay-buffer memory); scale to [0, 1].
        x = x.float() / 255.0
        x = self.features(x)
        V = self.V(x)
        A = self.A(x)
        return V + (A - A.mean(dim=-1, keepdim=True))

    @torch.no_grad()
    def get_action(self, state) -> int:
        """Greedy action for a single stacked observation of shape (4, 84, 84)."""
        device = next(self.parameters()).device
        state = torch.as_tensor(state, device=device).unsqueeze(0)
        return int(self(state).argmax(dim=-1).item())
