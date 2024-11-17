import argparse
import random
from itertools import count

import gymnasium as gym
import tomllib
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Logger, Task
from tqdm import tqdm

from models import DeepQNet
from utils import EpsilonScheduler, Memory


def load_params(config_path: str):
    global \
        episodes, \
        batch_size, \
        gamma, \
        tau, \
        lr, \
        epsilon_decay_steps, \
        epsilon_start, \
        epsilon_end, \
        memory_capacity

    with open(config_path, "rb") as f:
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

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    task = Task.init(
        project_name="Deep Q Learning/CartPole",
        task_name="Train",
        reuse_last_task_id=False,
    )
    config = load_params(args.config)
    task.connect(config)

    env = gym.make("CartPole-v1")
    state, _ = env.reset()

    train_net = DeepQNet(len(state), env.action_space.n).cuda()
    eval_net = DeepQNet(len(state), env.action_space.n).cuda()
    eval_net.load_state_dict(train_net.state_dict())

    epsilon = EpsilonScheduler(epsilon_start, epsilon_end, epsilon_decay_steps)
    optimizer = optim.AdamW(train_net.parameters(), lr=lr, amsgrad=True)
    criterion = nn.SmoothL1Loss()
    memory = Memory(memory_capacity, batch_size, len(state))

    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        prev_state = torch.tensor(state).cuda().unsqueeze(0)
        losses = []

        for t in count():
            prob_rand = epsilon.next()
            action = train_net.select_action(prev_state, prob_rand)

            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device="cuda")

            memory.push(prev_state, action.item(), observation, reward, terminated)
            prev_state = torch.tensor(observation).cuda().unsqueeze(0)

            if len(memory) >= batch_size:
                samples = random.sample(range(len(memory)), batch_size)

                batch = memory.sample(samples)

                state_q_values = train_net(batch["states"]).gather(1, batch["actions"])

                with torch.no_grad():
                    next_state_q_values = eval_net(batch["next_states"])
                    next_state_q_values[batch["terminated"]] = 0
                    next_state_q_values = next_state_q_values.max(1).values

                target_state_q_values = (next_state_q_values * gamma) + batch["rewards"]

                loss = criterion(state_q_values, target_state_q_values.unsqueeze(1))

                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(train_net.parameters(), 100)
                optimizer.step()

                eval_net.copy_weights(train_net, tau)

            if terminated or truncated:
                break

        Logger.current_logger().report_scalar("Episode length", "length", t, episode)
        Logger.current_logger().report_scalar("Epsilon", "epsilon", prob_rand, episode)
        loss = sum(losses) / len(losses) if losses else 0
        Logger.current_logger().report_scalar("Loss", "loss", loss, episode)
