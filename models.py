from random import randint, random

import numpy as np
import torch
from torch import nn


class DeepQNet(nn.Module):
    def __init__(self, state_size: int, action_space_size: int):
        super(DeepQNet, self).__init__()
        self.action_space_size = action_space_size

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_size),
        )

    def preprocess(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state).cuda().unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @torch.no_grad()
    def select_action(self, state: torch.Tensor, p_random: float) -> int:
        if random() > p_random:
            return self.network(state).max(1).indices.item()
        return randint(0, self.action_space_size - 1)

    def copy_weights(self, other: nn.Module, tau: float):
        other_sd = other.state_dict()
        self_sd = self.state_dict()

        for key in other_sd:
            self_sd[key] = other_sd[key] * tau + self_sd[key] * (1 - tau)

        self.load_state_dict(self_sd)


class DeepQConvNet(nn.Module):
    def __init__(self, action_space_size: int, stacked_frames: int):
        super().__init__()
        self.action_space_size = action_space_size

        self.network = nn.Sequential(
            nn.Conv2d(stacked_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3456, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_size),
        )

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        return torch.tensor(frame).to(torch.float32).cuda() / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @torch.no_grad()
    def select_action(self, state: torch.Tensor, p_random: float) -> int:
        if random() > p_random:
            return self.network(state.unsqueeze(0)).max(1).indices.item()
        return randint(0, self.action_space_size - 1)

    def copy_weights(self, other: nn.Module, tau: float):
        other_sd = other.state_dict()
        self_sd = self.state_dict()

        for key in other_sd:
            self_sd[key] = other_sd[key] * tau + self_sd[key] * (1 - tau)

        self.load_state_dict(self_sd)
