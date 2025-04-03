import numpy as np
import math
import random
from collections import namedtuple, deque

import gymnasium as gym
from gymnasium.spaces import flatten_space
from gymnasium.spaces.utils import flatten

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs/experiment_1')

from tqdm import tqdm
import matplotlib.pyplot as plt

from V1env import JustDoIt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is basically just a function approximater that takes the form V(s, w), where the weights are learned in the DQN.
# We are using a Q-Learning (deep) because off of my current intuition there's no sense of risk as of yet when it comes to certain actions taken place.
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()

        # self.conv1 = nn.Conv2d()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
         # This part is actually pretty weird, but the policy is what generates the Q-values for every possible action and then we sample using softmax
        return out

"""
We will implement the ReplayMemory class ->:

- This class will store the experiences (state, action, reward, next_state, done)
"""
Transition = namedtuple('Transition', ('State', 'Action', 'Reward', 'Next_State', 'Done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def train(policy_network, target_network,
          episodes, time_steps, batch_size, 
          epsilon, epsilon_decay, gamma, gamma_decay, criterion, optimizer):

    all_reward_sequences = []  # For graphing down the line
    replaysampler = ReplayMemory(capacity=1000)
    for episode in range(episodes):
        accumulated_reward = 0

        writer.add_scalar('Reward/Episode', accumulated_reward, episode)

        episode_reward = []  # Start an empty list for episode rewards
        
        # Reset the environment and get the initial observation; flatten it.
        start_state, _ = env.reset()
        current_state = flatten(env.observation_space, start_state)  # current_state is a flat numpy array

        for t in range(time_steps):  # Keep running until allocated steps are gone

            action_epsilon_chance = np.random.rand()

            if action_epsilon_chance < epsilon:
                # Exploration branch: sample a random action
                sampled_action = env.action_space.sample()
                next_state, reward, end, _, _ = env.step(sampled_action)
                next_state_flat = flatten(env.observation_space, next_state)
                # Use the flattened current state and next state in the transition
                transition = Transition(current_state, sampled_action, reward, next_state_flat, end)
                replaysampler.push(*transition)
            else:
                # Exploitation branch: choose action with the policy network
                with torch.no_grad():
                    current_state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device)
                    model_q_values = policy_network(current_state_tensor)
                    # Convert the tensor result to an integer action
                    action = int(torch.argmax(model_q_values).item())
                next_state, reward, end, _, _ = env.step(action)
                next_state_flat = flatten(env.observation_space, next_state)
                transition = Transition(current_state, action, reward, next_state_flat, end)
                replaysampler.push(*transition)

            # If we have enough samples, perform a training step.
            if(t % 10 == 0):
                if len(replaysampler) >= batch_size:
                    sampled_batch_experiences = replaysampler.sample(batch_size)
                    # Unpack the transitions into batches.
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*sampled_batch_experiences)
                    
                    batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=device)
                    batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=device)
                    batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
                    batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
                    batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=device)
                    
                    # Compute Q-values for current states and gather the Q-values for the actions taken.
                    q_values = policy_network(batch_states_tensor)
                    q_values = q_values.gather(1, batch_actions_tensor.unsqueeze(1)).squeeze(1)
                    
                    # Compute target Q-values using the target network.
                    with torch.no_grad():
                        next_q_values = target_network(batch_next_states_tensor)
                        max_next_q_values = next_q_values.max(1)[0]
                        target_q_values = batch_rewards_tensor + gamma * max_next_q_values * (1 - batch_dones_tensor)
                    
                    loss = criterion(q_values, target_q_values)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            accumulated_reward += reward
            episode_reward.append(accumulated_reward)
            
            # Update current state with the new flattened observation.
            current_state = next_state_flat

            if end:
                break

        epsilon *= epsilon_decay
        #gamma *= gamma_decay  Note: Typically gamma remains fixed; adjust as needed.

        all_reward_sequences.append(episode_reward)
        target_network.load_state_dict(policy_network.state_dict())
    writer.close()

    return policy_network, target_network, all_reward_sequences


""" 
Quick Review
<><><><><><><><><><><><><><>

This is a DQN model, and because of this we know we're trying to approximate the policy function instead of discretizing each
action and state to find some optimal policy and just converging to some optimal Q-table

Steps:
1. Initialize the env => training network, and the copied over target network
2. Initialize some episodic training loop
    - Grab rewards, iterate the states and actions (sampled with some epsilon factor (decay as well)) blah blah blah
3. Utilize an R+1 expectation state/action value methodology (n_step = 1 Temporal Difference Learning or MC, idk, just something with one step)
4. Apply the training of the policy to the network
5. Test that shit out lmfaooo
<><><><><><><><><><><><><><>
"""

# 1.) Setting Up the Environment & The dual Weilding DQN networks

env = JustDoIt()
action_dim = 4
flattened_obs_space = flatten_space(env.observation_space)
state_dim = flattened_obs_space.shape[0]

policy_net = DQN(state_dim=state_dim, action_dim=action_dim).to(device)
target_net = DQN(state_dim=state_dim, action_dim=action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())  # Copy the weights from the policy network to the target network

if __name__ == "__main__":
    # 2.) Training Loop
    episodes = 50
    time_steps = 500
    BATCH_SIZE = 64
    EPSILON = 1
    EPSILON_DECAY = 0.9998
    GAMMA = 0.75
    GAMMA_DECAY = 0.95
    LR = 1e-4
    CRITERION = nn.SmoothL1Loss()
    OPTIMIZER = optim.Adam(policy_net.parameters(), lr=LR)

    policy_network, target_network, all_reward_sequences = train(
        policy_network=policy_net,
        target_network=target_net,
        episodes=episodes,
        time_steps=time_steps,
        batch_size=BATCH_SIZE,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        gamma=GAMMA,
        gamma_decay=GAMMA_DECAY,
        criterion=CRITERION,
        optimizer=OPTIMIZER
    )


    def moving_average(data, window_size=10):
        """Compute the moving average with a given window size."""
        if len(data) < window_size:
            return data  # Not enough data points for a moving average
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



    def save_models():
        torch.save(target_network.state_dict(), 'target_state_dict.pth')
        torch.save(target_network, 'target_model.pth')
        torch.save(policy_network.state_dict(), 'policy_state_dict.pth')
        torch.save(policy_network, 'policy_network.pth')

    save_models()

    # Plot a scatter plot for each episode's moving average.
    plt.figure(figsize=(10, 6))
    for episode_idx, episode_rewards in enumerate(all_reward_sequences[len(all_reward_sequences) - 5:]):
        # Compute moving average with a window size of 10 (adjust as needed)
        ma = moving_average(episode_rewards, window_size=10)
        # Create x values corresponding to the time steps after computing the moving average
        x_vals = np.arange(len(ma))
        # Use scatter plot for this episode
        plt.scatter(x_vals, ma, label=f"Episode {episode_idx+1}", alpha=0.6)

    plt.xlabel("Time Step")
    plt.ylabel("Moving Average of Cumulative Reward")
    plt.title("Moving Average Scatter Plot for Each Episode")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("Rewards.png")