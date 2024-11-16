import math
import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

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
