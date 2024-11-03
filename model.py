import numpy as np
import torch
import torch.nn as nn


class DeepQ(nn.Module):
    def __init__(
        self,
        gamma: float = 0.99,
        stacked_frames: int = 4,
        actions_count: int = 4,
    ):
        super().__init__()

        self.gamma = gamma
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

    def forward(self, frames: np.ndarray, actions: np.ndarray):
        frames = torch.tensor(frames / 255.0, dtype=torch.float32)
        output = self.network(frames)
        return output
