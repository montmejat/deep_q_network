import argparse
import os
from datetime import datetime
from itertools import count

import tomllib
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Logger, Task
from tqdm import tqdm

from environment import Breakout, CartPole
from models import DeepQNet, DeepQConvNet
from utils import EpsilonScheduler, Memory, ImageMemory


def create_checkpoint_folder(name: str):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    folder = f"checkpoints/{name.replace(" ", "-")}_{datetime.now().isoformat()}"
    os.makedirs(folder)
    return folder


def load_params(config_path: str):
    global \
        episodes, \
        batch_size, \
        environment, \
        checkpoints, \
        gamma, \
        tau, \
        lr, \
        epsilon_warmup, \
        epsilon_decay_steps, \
        epsilon_start, \
        epsilon_end, \
        memory_capacity

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    episodes = config["general"]["episodes"]
    batch_size = config["general"]["batch_size"]
    environment = config["general"]["environment"]
    checkpoints = config["general"]["checkpoints"]
    gamma = config["general"]["gamma"]
    tau = config["general"]["tau"]
    lr = config["general"]["lr"]
    epsilon_warmup = config["epsilon"]["warmup"]
    epsilon_decay_steps = config["epsilon"]["decay_steps"]
    epsilon_start = config["epsilon"]["start"]
    epsilon_end = config["epsilon"]["end"]
    memory_capacity = config["memory"]["capacity"]

    return config


def log_actions(taken_actions: dict, last_n_actions: list, iteration: int):
    for taken_action in taken_actions:
        Logger.current_logger().report_scalar(
            title="Actions (%, total)",
            series=f"action '{taken_action}'",
            value=taken_actions[taken_action] / sum(taken_actions.values()),
            iteration=iteration,
        )

        Logger.current_logger().report_scalar(
            title="Actions (%, last 1000)",
            series=f"action '{taken_action}'",
            value=last_n_actions.count(taken_action) / len(last_n_actions),
            iteration=iteration,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config = load_params(args.config)
    folder = create_checkpoint_folder(environment)

    task = Task.init(
        project_name=f"Deep Q Learning/{environment}",
        task_name="Train",
        reuse_last_task_id=False,
    )
    task.connect(config)

    match environment:
        case "Cart Pole":
            env = CartPole()
            train_net = DeepQNet(4, env.action_space.n).cuda()
            eval_net = DeepQNet(4, env.action_space.n).cuda()
            memory = Memory(memory_capacity, batch_size, observation_size=4)

            state, _ = env.reset()

            action = env.action_space.sample()
            next_state, reward, terminated, _, _ = env.step(action)
            memory.push(state, action, next_state, reward, terminated)

        case "Breakout":
            env = Breakout()
            train_net = DeepQConvNet(env.action_space.n, stacked_frames=4).cuda()
            eval_net = DeepQConvNet(env.action_space.n, stacked_frames=4).cuda()
            memory = ImageMemory(
                env.action_space.n,
                batch_size,
                memory_capacity,
                stacked_frames=4,
                frame_shape=(105, 80),
            )

            state, _ = env.reset()

            action = 1  # Fire to start the game
            next_state, reward, terminated, _, _ = env.step(action)
            memory.push(state, action, next_state, reward, terminated)
            prev_state = next_state

            for _ in range(3):
                action = env.action_space.sample()
                next_state, reward, terminated, _, _ = env.step(action)
                memory.push(prev_state, action, next_state, reward, terminated)
                prev_state = next_state

        case _:
            raise ValueError(f"Unknown environment: {environment}")

    eval_net.load_state_dict(train_net.state_dict())

    epsilon = EpsilonScheduler(
        epsilon_start, epsilon_end, epsilon_decay_steps, epsilon_warmup
    )
    optimizer = optim.AdamW(train_net.parameters(), lr=lr, amsgrad=True)
    criterion = nn.SmoothL1Loss()

    for episode in tqdm(range(episodes)):
        prev_state = memory.last_states()
        losses = []

        for t in count():
            p_rand = epsilon.next()
            action = train_net.select_action(train_net.preprocess(prev_state), p_rand)

            next_state, reward, terminated, truncated, info = env.step(action)
            memory.push(prev_state[-1], action, next_state, reward, terminated)
            # memory.push(prev_state, action, next_state, reward, terminated)

            prev_state = memory.last_states()

            if len(memory) >= batch_size + 4:
                batch = memory.sample()

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
                state, _ = env.reset()
                action = 1  # Fire to start the game
                next_state, reward, terminated, _, _ = env.step(action)
                memory.push(state, action, next_state, reward, terminated)
                break

            if "lost_life" in info and info["lost_life"]:
                action = 1  # Fire to start the game
                next_state, reward, terminated, _, _ = env.step(action)
                memory.push(state, action, next_state, reward, terminated)
                break

        Logger.current_logger().report_scalar("Episode length", "length", t, episode)
        Logger.current_logger().report_scalar("Epsilon", "epsilon", p_rand, episode)
        loss = sum(losses) / len(losses) if losses else 0
        Logger.current_logger().report_scalar("Loss", "loss", loss, episode)
        log_actions(env.taken_actions, env.last_n_actions, episode)

        if (episode + 1) % checkpoints == 0:
            torch.save(train_net.state_dict(), f"{folder}/checkpoint_{episode + 1}.pth")
