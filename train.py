import argparse
import collections
import os
import random
from datetime import datetime

import ale_py  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from clearml import Logger, Task
from PIL import Image
from tqdm import tqdm

from model import DeepQNet
from utils import EpsilonScheduler, Memory


def create_checkpoint_folder():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    folder = f"checkpoints/{datetime.now().isoformat()}"
    os.makedirs(folder)

    return folder


def log_epsilon(epsilon: float, iteration: int):
    Logger.current_logger().report_scalar(
        title="parameters", series="epsilon", value=epsilon, iteration=iteration
    )


def log_loss(loss: float, iteration: int):
    Logger.current_logger().report_scalar(
        title="model", series="loss", value=loss, iteration=iteration
    )


def log_reward(reward: float, iteration: int):
    Logger.current_logger().report_scalar(
        title="environment", series="reward", value=reward, iteration=iteration
    )


def log_actions(
    taken_actions: dict, last_1000_actions: collections.deque, iteration: int
):
    for taken_action in taken_actions:
        Logger.current_logger().report_scalar(
            title="actions (%)",
            series=f"action '{taken_action}'",
            value=taken_actions[taken_action] / sum(taken_actions.values()),
            iteration=iteration,
        )

        Logger.current_logger().report_scalar(
            title="actions (%, last 1000)",
            series=f"action '{taken_action}'",
            value=last_1000_actions.count(taken_action) / len(last_1000_actions),
            iteration=iteration,
        )


def log_debug_samples(batch_size: int, batch: dict, iteration: int):
    for i in range(batch_size):
        frames = batch["frames"][i]
        frames = (
            (torch.hstack([frame * 255 for frame in frames])).numpy().astype(np.uint8)
        )
        image = Image.fromarray(frames)

        Logger.current_logger().report_image(
            title="Training frames",
            series=f"Batch {i}",
            iteration=iteration,
            image=image,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5_000_000)
    parser.add_argument("--epsilon-iterations", type=int, default=500_000)
    parser.add_argument("--epsilon-min-value", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-at", type=int, default=100_000)
    parser.add_argument("--log-frequency", type=int, default=30)
    parser.add_argument("--log-debug-samples", action="store_true")
    parser.add_argument("--memory-capacity", type=int, default=100_000)
    args = parser.parse_args()

    task = Task.init(project_name="Deep Q Learning", task_name="Train")

    folder = create_checkpoint_folder()

    env = gym.make("ALE/Breakout-v5")
    _, info = env.reset(seed=0)
    lives = info["lives"]

    scheduler = EpsilonScheduler(
        stop_at=args.epsilon_iterations,
        min_value=args.epsilon_min_value,
    )
    memory = Memory(
        batch_size=args.batch_size,
        action_count=env.action_space.n,
        capacity=args.memory_capacity,
    )

    model = DeepQNet(actions_count=env.action_space.n)
    model.train()
    model.cuda()

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.00025, alpha=0.95, eps=0.01
    )
    criterion = torch.nn.MSELoss()

    done, truncated = True, True

    taken_actions = {i: 0 for i in range(env.action_space.n)}
    last_1000_actions = collections.deque(maxlen=5_000)
    loss_values = []
    reward_values = []

    for iteration in tqdm(range(1, args.iterations + 1)):
        epsilon = scheduler(iteration)

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                frames = memory.last_frames()
                if frames is not None:
                    output = model(frames[None, :].cuda())
                    action = output.argmax().item()
                else:
                    action = env.action_space.sample()

        taken_actions[action] += 1
        last_1000_actions.append(action)

        frame, reward, done, truncated, info = env.step(action)
        memory.append(action, torch.tensor(frame), reward, done)

        if memory.ready_to_sample and iteration % args.batch_size == 0:
            batch = memory.sample()

            optimizer.zero_grad()

            pred_q_values = model(batch["next_frames"].cuda())
            pred_q_values[batch["dones"]] = 0
            rewards = torch.sign(batch["rewards"]).cuda()
            # rewards = batch["rewards"].cuda()
            target_q_values = rewards + 0.99 * pred_q_values.max(dim=1).values

            loss = criterion(
                model(batch["frames"].cuda())
                .gather(1, batch["actions"].cuda().unsqueeze(1))
                .squeeze(1),
                target_q_values,
            )

            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            reward_values.append(reward)

            if iteration % (10 * args.batch_size) == 0:
                log_epsilon(epsilon, iteration)
                log_actions(taken_actions, last_1000_actions, iteration)

                loss = sum(loss_values) / len(loss_values)
                loss_values = []
                log_loss(loss, iteration)

                reward = sum(reward_values) / len(reward_values)
                reward_values = []
                log_reward(reward, iteration)

                if args.log_debug_samples:
                    log_debug_samples(args.batch_size, batch, iteration)

        if iteration % args.checkpoint_at == 0:
            state_dict = model.state_dict()
            torch.save(state_dict, f"{folder}/checkpoint_{iteration}.pth")

        if done or truncated:
            env.reset()

        if info["lives"] < lives:
            lives = info["lives"]
            env.step(1)

    env.close()
