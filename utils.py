import math
import random
from collections import deque, namedtuple
from itertools import islice
from typing import Tuple

import numpy as np
import torch

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminated")
)
ImageTransition = namedtuple(
    "ImageTransition", ("state", "action", "next_state", "reward", "terminated")
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


class ImageMemory:
    def __init__(
        self,
        action_count: int,
        batch_size: int,
        capacity: int,
        stacked_frames: int,
        frame_shape: Tuple[int, int],
    ):
        self.action_count = action_count
        self.batch_size = batch_size
        self.stacked_frames = stacked_frames
        self.frame_shape = frame_shape
        self.memory = deque(maxlen=capacity)

    def __stack_states(self, index: int):
        slice = list(islice(self.memory, index + 1 - self.stacked_frames, index + 1))
        return np.stack([item.state for item in slice], axis=0)

    def __stack_next_states(self, index: int):
        slice = list(islice(self.memory, index + 1 - self.stacked_frames, index + 1))
        return np.stack([item.next_state for item in slice], axis=0)

    def push(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        terminated: bool,
    ):
        if state.shape != self.frame_shape:
            state = np.mean(state, axis=-1)[::2, ::2].astype(np.uint8)
        if next_state.shape != self.frame_shape:
            next_state = np.mean(next_state, axis=-1)[::2, ::2].astype(np.uint8)

        self.memory.append(
            ImageTransition(state, action, next_state, reward, terminated)
        )

    def sample(self):
        frames_dim = (self.batch_size, self.stacked_frames, *self.frame_shape)

        states = torch.empty(frames_dim)
        actions = torch.empty((self.batch_size, 1), dtype=torch.int64)
        next_states = torch.empty(frames_dim)
        rewards = torch.empty(self.batch_size)
        terminated = []

        for i, rand_index in enumerate(
            random.sample(
                range(self.stacked_frames - 1, len(self.memory)), self.batch_size
            )
        ):
            states[i] = torch.tensor(self.__stack_states(rand_index))
            actions[i] = self.memory[rand_index].action
            next_states[i] = torch.tensor(self.__stack_next_states(rand_index))
            rewards[i] = self.memory[rand_index].reward
            terminated.append(self.memory[rand_index].terminated)

        return {
            "states": states.cuda(),
            "actions": actions.cuda(),
            "next_states": next_states.cuda(),
            "rewards": rewards.cuda(),
            "terminated": terminated,
        }

    def last_states(self):
        return self.__stack_next_states(len(self.memory) - 1)

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
