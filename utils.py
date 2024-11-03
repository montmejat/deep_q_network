import itertools
import random
from collections import deque

import numpy as np


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
        capacity: int = 10_000,
    ):
        self.stacked_frames = stacked_frames
        self.action_count = action_count
        self.batch_size = batch_size
        self.memory = deque(maxlen=capacity)

    def __to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        return np.mean(frame, axis=-1).astype(np.uint8)

    def __downscale(self, frame: np.ndarray) -> np.ndarray:
        return frame[::2, ::2]

    def __stack_slice(self, index: int):
        return np.stack(
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

    def append(self, action: np.ndarray, frame: np.ndarray, reward: float, done: bool):
        frame = self.__to_grayscale(frame)
        frame = self.__downscale(frame)
        self.memory.append((action, frame, reward, done))

    def sample(self):
        actions = []
        frames = []
        rewards = []
        dones = []
        next_frames = []

        for _ in range(self.batch_size):
            rand_index = random.randint(self.stacked_frames - 1, len(self.memory) - 2)

            action = self.memory[rand_index][0]
            action_hot = np.zeros((self.action_count))
            action_hot[action] = 1

            actions.append(action_hot)
            frames.append(self.__stack_slice(rand_index))
            rewards.append(self.memory[rand_index][2])
            dones.append(self.memory[rand_index][3])
            next_frames.append(self.__stack_slice(rand_index + 1))

        return {
            "actions": np.array(actions),
            "frames": np.array(frames),
            "rewards": rewards,
            "dones": dones,
            "next_frames": np.array(next_frames),
        }
