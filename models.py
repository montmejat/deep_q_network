from random import randint, random

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

    def forward(self, x: torch.Tensor):
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
