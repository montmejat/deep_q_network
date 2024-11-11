import itertools
import random
from collections import deque
from typing import Tuple

import numpy as np
import torch


class EpsilonScheduler:
    def __init__(self, stop_at: int, min_value: float):
        self.stop_at = stop_at
        self.min_value = min_value

    def __call__(self, iteration: int):
        return max(1 - iteration / self.stop_at, self.min_value)


class Memory:
    def __init__(
        self,
        stacked_frames: int = 4,
        action_count: int = 4,
        batch_size: int = 32,
        capacity: int = 20_000,
        frame_shape: Tuple[int, int] = (105, 80),
    ):
        self.stacked_frames = stacked_frames
        self.action_count = action_count
        self.batch_size = batch_size
        self.frame_shape = frame_shape
        self.memory = deque(maxlen=capacity)

    def __to_grayscale(self, frame: torch.Tensor) -> torch.Tensor:
        return torch.mean(frame.float(), axis=-1).to(torch.uint8)

    def __downscale(self, frame: torch.Tensor) -> torch.Tensor:
        return frame[::2, ::2]

    def __stack_slice(self, index: int):
        return torch.stack(
            [
                frame[1]
                for frame in list(
                    itertools.islice(
                        self.memory,
                        index - (self.stacked_frames - 1),
                        index + 1,
                    )
                )
            ],
            axis=0,
        )

    @property
    def ready_to_sample(self) -> bool:
        return len(self.memory) >= self.batch_size

    def append(
        self, action: np.ndarray, frame: torch.Tensor, reward: float, done: bool
    ):
        frame = self.__to_grayscale(frame)
        frame = self.__downscale(frame)
        self.memory.append((action, frame, reward, done))

    def sample(self):
        frames_dim = (self.batch_size, self.stacked_frames, *self.frame_shape)
        actions_hot = torch.empty((self.batch_size, self.action_count))
        actions = torch.empty(self.batch_size, dtype=torch.int64)
        frames = torch.empty(frames_dim)
        rewards = []
        dones = []
        next_frames = torch.empty(frames_dim)

        for i in range(self.batch_size):
            rand_index = random.randint(self.stacked_frames - 1, len(self.memory) - 2)

            action = self.memory[rand_index][0]
            action_hot = torch.zeros((self.action_count))
            action_hot[action] = 1

            actions_hot[i] = action_hot
            actions[i] = action
            frames[i] = self.__stack_slice(rand_index)
            rewards.append(self.memory[rand_index][2])
            dones.append(self.memory[rand_index][3])
            next_frames[i] = self.__stack_slice(rand_index + 1)

        return {
            "actions_hot": actions_hot,
            "actions": actions,
            "frames": frames / 255.0,
            "rewards": torch.tensor(rewards),
            "dones": dones,
            "next_frames": next_frames / 255.0,
        }

    def last_frames(self):
        if len(self.memory) < self.stacked_frames:
            return None

        return self.__stack_slice(len(self.memory) - 1) / 255.0
