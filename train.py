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


def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device="cuda",
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device="cuda")
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


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

    task = Task.init(
        project_name="Deep Q Learning/CartPole",
        task_name="Train",
        reuse_last_task_id=False,
    )
    task.connect(config)

    env = gym.make("CartPole-v1")

    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DeepQNet(n_observations, n_actions).cuda()
    target_net = DeepQNet(n_observations, n_actions).cuda()
    target_net.load_state_dict(policy_net.state_dict())

    epsilon = EpsilonScheduler(epsilon_start, epsilon_end, epsilon_decay_steps)
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    episode_durations = []

    for episode in tqdm(range(episodes)):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)

        for t in count():
            prob_rand = epsilon.next()

            action = policy_net.select_action(state, prob_rand)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device="cuda")
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device="cuda"
                ).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * tau + target_net_state_dict[key] * (1 - tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                break

        Logger.current_logger().report_scalar("Episode length", "length", t, episode)
        Logger.current_logger().report_scalar("Epsilon", "epsilon", prob_rand, episode)
