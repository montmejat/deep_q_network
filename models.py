from torch import nn


class DQN(nn.Module):
    def __init__(self, state_size: int, actions_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions_size),
        )

    def forward(self, x):
        return self.network(x)
