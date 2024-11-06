import argparse
import random

import ale_py  # noqa: F401
import gymnasium as gym
import torch

from model import DeepQNet
from utils import EpsilonScheduler, Memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--epsilon-iterations", type=int, default=5_000)
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

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            frames = memory.last_frames()
            if frames is not None:
                output = model(frames[None, :])
                action = output.argmax().item()
            else:
                action = env.action_space.sample()

        frame, reward, done, truncated, info = env.step(action)
        memory.append(action, torch.tensor(frame), reward, done)

        if memory.ready_to_sample and iteration % args.batch_size == 0:
            batch = memory.sample()

            optimizer.zero_grad()

            pred_q_values = model(batch["next_frames"])
            pred_q_values[batch["dones"]] = 0
            rewards = torch.sign(batch["rewards"])
            target_q_values = rewards + 0.99 * pred_q_values.max(dim=1).values

            loss = criterion(
                target_q_values,
                model(batch["frames"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1),
            )

            loss.backward()
            optimizer.step()

            print(f"Iteration: {iteration}, Loss: {loss.item()}")
    
        if done or truncated:
            env.reset()

    model.eval()
    memory = Memory(batch_size=1, action_count=env.action_space.n)

    env = gym.make("ALE/Breakout-v5", render_mode="human")

    for i in range(1):
        env.reset()

        frame, _, _, _, _ = env.step(0)
        memory.append(0, torch.tensor(frame), 0, False)

        done = False

        while not done:
            frames = memory.last_frames()

            action = 0
            if frames is not None:
                output = model(frames[None, :])
                action = output.argmax().item()
                print("action:", action)

            frame, _, done, _, _ = env.step(action)
            memory.append(0, torch.tensor(frame), 0, False)

            env.render()

    env.close()
