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
    def select_action(self, state: torch.Tensor, p_random: float):
        if random() > p_random:
            return self.network(state).max(1).indices.view(1, 1)
        return torch.tensor([[randint(0, self.action_space_size - 1)]]).cuda()
