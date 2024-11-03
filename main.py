import argparse
import random

import ale_py  # noqa: F401
import gymnasium as gym
from model import DeepQ
from utils import EpsilonScheduler, Memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--epsilon-iterations", type=int, default=100)
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

    model = DeepQ(actions_count=env.action_space.n)
    model.train()

    for iteration in range(args.iterations):
        epsilon = scheduler(iteration)
        action = env.action_space.sample() if random.random() < epsilon else 0

        frame, reward, done, truncated, info = env.step(action)
        memory.append(action, frame, reward, done)

        if memory.ready_to_sample and iteration % args.batch_size == 0:
            batch = memory.sample()
            output = model(batch["frames"], batch["actions"])

    env.close()
