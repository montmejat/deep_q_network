import argparse
from itertools import count

import gymnasium as gym
import tomllib
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Logger, Task
from tqdm import tqdm

from models import DeepQNet
from utils import EpsilonScheduler, ReplayMemory, Transition


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    episodes = config["general"]["episodes"]
    batch_size = config["general"]["batch_size"]
    gamma = config["general"]["gamma"]
    tau = config["general"]["tau"]
    lr = config["general"]["lr"]
    epsilon_decay_steps = config["epsilon"]["decay_steps"]
    epsilon_start = config["epsilon"]["start"]
    epsilon_end = config["epsilon"]["end"]
    memory_capacity = config["memory"]["capacity"]

    task = Task.init(
        project_name="Deep Q Learning/CartPole",
        task_name="Train",
        reuse_last_task_id=False,
    )
    task.connect(config)

    env = gym.make("CartPole-v1")
    state, _ = env.reset()

    train_net = DeepQNet(len(state), env.action_space.n).cuda()
    eval_net = DeepQNet(len(state), env.action_space.n).cuda()
    eval_net.load_state_dict(train_net.state_dict())

    epsilon = EpsilonScheduler(epsilon_start, epsilon_end, epsilon_decay_steps)
    optimizer = optim.AdamW(train_net.parameters(), lr=lr, amsgrad=True)
    criterion = nn.SmoothL1Loss()
    memory = ReplayMemory(memory_capacity)

    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        state = torch.tensor(state).cuda().unsqueeze(0)
        losses = []

        for t in count():
            prob_rand = epsilon.next()
            action = train_net.select_action(state, prob_rand)

            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device="cuda")
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device="cuda"
                ).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state

            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)),
                    dtype=torch.bool,
                ).cuda()
                non_final_next_states = torch.cat(
                    [s for s in batch.next_state if s is not None]
                )
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                state_action_values = train_net(state_batch).gather(1, action_batch)

                next_state_values = torch.zeros(batch_size, device="cuda")
                with torch.no_grad():
                    next_state_values[non_final_mask] = (
                        eval_net(non_final_next_states).max(1).values
                    )
                expected_state_action_values = (
                    next_state_values * gamma
                ) + reward_batch

                loss = criterion(
                    state_action_values, expected_state_action_values.unsqueeze(1)
                )
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(train_net.parameters(), 100)
                optimizer.step()

                eval_net.copy_weights(train_net, tau)

            if done:
                break

        Logger.current_logger().report_scalar("Episode length", "length", t, episode)
        Logger.current_logger().report_scalar("Epsilon", "epsilon", prob_rand, episode)
        loss = sum(losses) / len(losses) if losses else 0
        Logger.current_logger().report_scalar("Loss", "loss", loss, episode)
