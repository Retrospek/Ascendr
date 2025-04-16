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

import itertools

from V1env import JustDoIt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is basically just a function approximater that takes the form V(s, w), where the weights are learned in the DQN.
# We are using a Q-Learning (deep) because off of my current intuition there's no sense of risk as of yet when it comes to certain actions taken place.
class DQN(nn.Module):
    
    def __init__(self, state_dim, action_dim, gridDim):
        # DQN Model
        # -------------------------------------------
        # Args:
        # - state_dim (int): Dimension of flattened state space
        # - action_dim (int): Number of discrete actions
        # - gridDim (int): Width/height of square image grid (e.g., 50)
        # -------------------------------------------
        super(DQN, self).__init__()

        self.gridDim = gridDim

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.convfc1 = nn.Linear(800, 512)
        self.convfc2 = nn.Linear(512, 256)
        self.convfc3 = nn.Linear(256, 128)
        self.convfc4_solo = nn.Linear(128, action_dim)

        self.fc1 = nn.Linear(state_dim - gridDim ** 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3_solo = nn.Linear(64, action_dim)

        self.combined_pred = nn.Linear(128, action_dim)

        self.relu = nn.ReLU()

    def forward(self, input):
        batch_size = input.size(0)

        image = input[:, :self.gridDim * self.gridDim].view(batch_size, 1, self.gridDim, self.gridDim)

        """
        dense = input[:, self.gridDim * self.gridDim:]

        lin = self.relu(self.fc1(dense))
        lin = self.relu(self.fc2(lin))
        """

        conv = self.pool(self.relu(self.conv1(image)))
        conv = self.pool2(self.relu(self.conv2(conv)))
        conv = torch.flatten(conv, start_dim=1)
        conv = self.relu(self.convfc1(conv))
        conv = self.relu(self.convfc2(conv))
        conv = self.relu(self.convfc3(conv))

        output = self.relu(self.convfc4_solo(conv))
        print(output.shape)
        #combined_output = torch.cat((conv, lin), dim=1)
        #output = self.combined_pred(combined_output)
        return output


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
          episodes, batch_size, 
          epsilon, epsilon_end, epsilon_decay, gamma, criterion, optimizer,
          gridDim, env):

    all_reward_sequences = []  # For graphing down the line
    replaysampler = ReplayMemory(capacity=1500)
    for episode in range(episodes):
        accumulated_reward = 0

        writer.add_scalar('Reward/Episode', accumulated_reward, episode)

        episode_reward = []  # Start an empty list for episode rewards
        
        # Reset the environment and get the initial observation; flatten it.
        start_state, _ = env.reset()
        current_state = flatten(env.observation_space, start_state)  # current_state is a flat numpy array

        for t in range(400):  # Keep running until allocated energy is gone
            
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
                    current_state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
                    model_q_values = policy_network(current_state_tensor)
                    # Convert the tensor result to an integer action
                    action = int(torch.argmax(model_q_values).item())
                next_state, reward, end, _, _ = env.step(action)
                next_state_flat = flatten(env.observation_space, next_state)
                transition = Transition(current_state, action, reward, next_state_flat, end)
                replaysampler.push(*transition)

            # If we have enough samples, perform a training step.
            #if(t % 5 == 0):
            if len(replaysampler) >= batch_size:
                sampled_batch_experiences = replaysampler.sample(batch_size)
                # Unpack the transitions into batches.
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*sampled_batch_experiences)
                
                batch_states_tensor = torch.tensor(np.stack(batch_states), dtype=torch.float32, device=device)
                batch_actions_tensor = torch.tensor(np.stack(batch_actions), dtype=torch.long, device=device)
                batch_rewards_tensor = torch.tensor(np.stack(batch_rewards), dtype=torch.float32, device=device)
                batch_next_states_tensor = torch.tensor(np.stack(batch_next_states), dtype=torch.float32, device=device)
                batch_dones_tensor = torch.tensor(np.stack(batch_dones), dtype=torch.float32, device=device)
                
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

        epsilon = epsilon_end + (epsilon - epsilon_end) * math.exp(-1. * t / epsilon_decay)
        """
        EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
        """
        #gamma *= gamma_decay  Note: Typically gamma remains fixed; adjust as needed.

        all_reward_sequences.append(episode_reward)
        target_network.load_state_dict(policy_network.state_dict())
    writer.close()

    return policy_network, target_network, all_reward_sequences

def moving_average(data, window_size=10):
        """Compute the moving average with a given window size."""
        if len(data) < window_size:
            return data  # Not enough data points for a moving average
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

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

#print(f"State Dim: {state_dim}")
#print(f"Action Dim: {action_dim}")

policy_net = DQN(state_dim=state_dim, action_dim=action_dim, gridDim=30).to(device)
target_net = DQN(state_dim=state_dim, action_dim=action_dim, gridDim=30).to(device)
target_net.load_state_dict(policy_net.state_dict())  # Copy the weights from the policy network to the target network

if __name__ == "__main__":
    # 2.) Training Loop
    episodes = 75
    BATCH_SIZE = 32
    GAMMA = 0.7
    EPSILON_START = 0.995
    EPSILON_END = 0.135
    EPSILON_DECAY = 800
    LR = 1.1e-3
    CRITERION = nn.SmoothL1Loss()
    OPTIMIZER = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    policy_network, target_network, all_reward_sequences = train(
        policy_network=policy_net, target_network=target_net,
        episodes=episodes,
        batch_size=BATCH_SIZE,
        epsilon=EPSILON_START, epsilon_end = EPSILON_END, epsilon_decay=EPSILON_DECAY,
        gamma=GAMMA,
        criterion=CRITERION, optimizer=OPTIMIZER,
        gridDim=50,
        env = JustDoIt()
    )

    def save_models():
        torch.save(target_network.state_dict(), 'target_state_dict.pth')
        torch.save(target_network, 'target_model.pth')
        torch.save(policy_network.state_dict(), 'policy_state_dict.pth')
        torch.save(policy_network, 'policy_network.pth')

    save_models()

    plt.figure(figsize=(12, 8))
    # Use a colormap to generate distinct colors for each episode
    num_episodes = 40
    colors = plt.cm.viridis(np.linspace(0, 1, num_episodes))

    # Plot the moving average for the last 40 episodes using line graphs
    for episode_idx, episode_rewards in enumerate(all_reward_sequences[-num_episodes:]):
        ma = moving_average(episode_rewards, window_size=10)
        x_vals = np.arange(len(ma))
        plt.plot(x_vals, ma, label=f"Episode {episode_idx+1}", color=colors[episode_idx], linewidth=2)

    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel("Moving Average of Cumulative Reward", fontsize=14)
    plt.title("Moving Average Line Plot for Each Episode", fontsize=16)
    plt.legend(loc="upper left", fontsize=10, ncol=2)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig("Rewards.png", dpi=300)
    plt.show()