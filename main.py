import argparse
import random

import ale_py  # noqa: F401
import gymnasium as gym
import torch

from model import DeepQNet
from utils import EpsilonScheduler, Memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--epsilon-iterations", type=int, default=1_000)
    parser.add_argument("--epsilon-min-value", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    env = gym.make("ALE/Breakout-v5")
    info = env.reset(seed=0)

    scheduler = EpsilonScheduler(
        stop_at=args.epsilon_iterations,
        min_value=args.epsilon_min_value,
    )
    memory = Memory(batch_size=args.batch_size, action_count=env.action_space.n)

    model = DeepQNet(actions_count=env.action_space.n)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for iteration in range(args.iterations):
        epsilon = scheduler(iteration)
        action = env.action_space.sample() if random.random() < epsilon else 0

        frame, reward, done, truncated, info = env.step(action)
        memory.append(action, torch.tensor(frame), reward, done)

        if memory.ready_to_sample and iteration % args.batch_size == 0:
            batch = memory.sample()

            optimizer.zero_grad()

            pred_q_values = model(batch["frames"], batch["actions"])

            next_q_values = model(batch["next_frames"], batch["actions"])
            next_q_values[batch["dones"]] = 0
            rewards = torch.sign(batch["rewards"])
            q_values = rewards + 0.99 * next_q_values.max(axis=1).values

            loss = criterion(pred_q_values, batch["actions"] * q_values[:, None])

            loss.backward()
            optimizer.step()

            print(f"Iteration: {iteration}, Loss: {loss.item()}")

    env.close()
