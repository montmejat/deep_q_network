import os
from datetime import datetime
from itertools import count

import ale_py  # noqa: F401
import gymnasium as gym
import torch

from environment import Breakout
from models import DeepQConvNet
from utils import ImageMemory

if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5")

    checkpoints = os.listdir("checkpoints")
    checkpoints = [
        datetime.fromisoformat(checkpoint.replace("Breakout_", ""))
        for checkpoint in checkpoints
        if "Breakout_" in checkpoint
    ]
    checkpoints.sort()
    checkpoint_folder = f"Breakout_{checkpoints[-1].isoformat()}"

    checkpoints = os.listdir(f"checkpoints/{checkpoint_folder}")
    checkpoints.sort(key=lambda x: int(x.split("_")[1][:-4]))

    model = DeepQConvNet(env.action_space.n, stacked_frames=4)
    checkpoint = f"checkpoints/{checkpoint_folder}/{checkpoints[-1]}"
    print("Loading checkpoint:", checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval().cuda()

    memory = ImageMemory(
        env.action_space.n,
        1,
        100,
        stacked_frames=4,
        frame_shape=(105, 80),
    )

    env = Breakout(render_mode="human")

    prev_state, _ = env.reset()

    for _ in range(3):
        action = 0
        next_state, reward, terminated, _, _ = env.step(action)
        memory.push(prev_state, action, next_state, reward, terminated)
        prev_state = next_state

    action = 1
    next_state, reward, terminated, _, _ = env.step(action)
    memory.push(prev_state, action, next_state, reward, terminated)
    prev_state = memory.last_states()

    for i in range(10):
        for t in count():
            action = model.select_action(model.preprocess(prev_state), p_random=0)
            next_state, reward, terminated, truncated, info = env.step(action)
            memory.push(prev_state[-1], action, next_state, reward, terminated)
            prev_state = memory.last_states()

            env.render()

            if info["lost_life"]:
                env.step(1)
                break

            if terminated:
                break

        print(f"Episode {i} finished after {t} steps")
