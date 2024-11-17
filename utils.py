import random
import math
from collections import deque, namedtuple

import numpy as np
import torch

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminated")
)


class Memory:
    def __init__(self, capacity: int, batch_size: int, observation_size: int):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.observation_size = observation_size

    def push(
        self,
        state: torch.Tensor,
        action: int,
        next_state: np.ndarray,
        reward: float,
        terminated: bool,
    ):
        next_state = torch.tensor(next_state).cuda().unsqueeze(0)
        self.memory.append(Transition(state, action, next_state, reward, terminated))

    def sample(self):
        actions = torch.empty((self.batch_size, 1), dtype=torch.int64)
        states = torch.empty((self.batch_size, self.observation_size))
        next_states = torch.empty((self.batch_size, self.observation_size))
        rewards = torch.empty(self.batch_size)
        terminated = []

        for i, rand_index in enumerate(
            random.sample(range(len(self.memory)), self.batch_size)
        ):
            states[i] = self.memory[rand_index].state
            actions[i] = self.memory[rand_index].action
            next_states[i] = self.memory[rand_index].next_state
            rewards[i] = self.memory[rand_index].reward
            terminated.append(self.memory[rand_index].terminated)

        return {
            "states": states.cuda(),
            "actions": actions.cuda(),
            "next_states": next_states.cuda(),
            "rewards": rewards.cuda(),
            "terminated": terminated,
        }

    def __len__(self):
        return len(self.memory)


class EpsilonScheduler:
    def __init__(self, start: int, end: int, decay_steps: int):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.iteration = 0

    def next(self):
        epsilon = self.end + (self.start - self.end) * math.exp(
            -1.0 * self.iteration / self.decay_steps
        )
        self.iteration += 1
        return epsilon
