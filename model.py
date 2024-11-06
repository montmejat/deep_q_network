import torch
import torch.nn as nn


class DeepQNet(nn.Module):
    def __init__(
        self,
        stacked_frames: int = 4,
        actions_count: int = 4,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(stacked_frames, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2816, 256),
            nn.ReLU(),
            nn.Linear(256, actions_count),
        )

    def forward(self, frames: torch.Tensor, actions: torch.Tensor = None):
        if actions is None:
            return self.network(frames)

        output = self.network(frames)
        return output * actions
