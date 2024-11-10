import os
from datetime import datetime

import ale_py  # noqa: F401
import gymnasium as gym
import torch

from model import DeepQNet
from utils import Memory

if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5")

    checkpoints = os.listdir("checkpoints")
    checkpoints = [datetime.fromisoformat(checkpoint) for checkpoint in checkpoints]
    checkpoints.sort()
    checkpoint_folder = checkpoints[-1].isoformat()

    checkpoints = os.listdir(f"checkpoints/{checkpoint_folder}")
    checkpoints.sort(key=lambda x: int(x.split("_")[1][:-4]))

    model = DeepQNet(actions_count=env.action_space.n)
    checkpoint = f"checkpoints/{checkpoint_folder}/{checkpoints[-1]}"
    print("Loading checkpoint:", checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()

    memory = Memory(batch_size=1, action_count=env.action_space.n)

    env = gym.make("ALE/Breakout-v5", render_mode="human")

    for i in range(10):
        _, info = env.reset()
        lives = info["lives"]

        env.step(1)

        done = False
        while not done:
            frames = memory.last_frames()

            action = 0
            if frames is not None:
                output = model(frames[None, :])
                action = output.argmax().item()

            frame, reward, done, _, info = env.step(action)
            memory.append(action, torch.tensor(frame), 0, False)

            env.render()

            if info["lives"] < lives:
                lives = info["lives"]
                env.step(1)
