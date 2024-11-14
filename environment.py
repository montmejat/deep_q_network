from collections import deque

import ale_py  # noqa: F401
import gymnasium as gym


class CartPole:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action: int):
        frame, reward, done, truncated, info = self.env.step(action)
        return frame, reward, done, False, truncated, info


class Breakout:
    def __init__(self):
        self.env = gym.make("ALE/Breakout-v5")
        self.action_space = self.env.action_space
        self.taken_actions = {i: 0 for i in range(self.action_space.n)}
        self.last_n_actions = deque(maxlen=1000)

    def reset(self):
        state, info = self.env.reset()
        self.lives = info["lives"]
        return state, info

    def step(self, action: int):
        frame, reward, done, truncated, info = self.env.step(action)

        self.taken_actions[action] += 1
        self.last_n_actions.append(action)

        lost_life = False
        if self.lives > info["lives"]:
            self.lives = info["lives"]
            lost_life = True

        return frame, reward, done, lost_life, truncated, info
