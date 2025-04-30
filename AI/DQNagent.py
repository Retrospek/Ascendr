import numpy as np
import math
import random
from collections import namedtuple, deque

import gymnasium as gym
from gymnasium.spaces import flatten_space
from gymnasium.spaces.utils import flatten

# Model Include
from DQNmodels import UNOarm_sign_based, UNOarm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter(log_dir='./runs/experiment_1')

from tqdm import tqdm
import matplotlib.pyplot as plt

import itertools

from v1_1arm.V1env import JustDoItV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
          env):

    episodic_rewards = []  # For graphing down the line
    replaysampler = ReplayMemory(capacity=1500)
    for episode in range(episodes):
        accumulated_reward = 0

        #writer.add_scalar('Reward/Episode', accumulated_reward, episode)
        
        start_state, _ = env.reset()
        current_state = flatten(env.observation_space, start_state)  # current_state is a flat numpy array
        for t in range(500):  
            action_epsilon_chance = np.random.rand()

            if action_epsilon_chance < epsilon:
                sampled_action = env.action_space.sample()
                next_state, reward, end, _, _ = env.step(sampled_action)
                next_state_flat = flatten(env.observation_space, next_state)
                transition = Transition(current_state, sampled_action, reward, next_state_flat, end)
                replaysampler.push(*transition)
            else:
                # Exploitation branch: choose action with the policy network
                with torch.no_grad():
                    current_state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
                    model_q_values = policy_network(current_state_tensor)
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
            
            # Update current state with the new flattened observation.
            current_state = next_state_flat

            if end:
                break
        
        print(f"EPISODE: {episode}, EPSILON: {epsilon}, REWARD_TOTAL(eps): {accumulated_reward}\n")

        epsilon = epsilon_end + (epsilon - epsilon_end) * math.exp(-1. * t / epsilon_decay)
        episodic_rewards.append(accumulated_reward)

        #gamma *= gamma_decay  Note: Typically gamma remains fixed; adjust as needed.
        if(episode%10 == 0):
            target_network.load_state_dict(policy_network.state_dict())
    #writer.close()

    return policy_network, target_network, episodic_rewards

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

env = JustDoItV1()
action_dim = 4
flattened_obs_space = flatten_space(env.observation_space)
state_dim = flattened_obs_space.shape[0]

#print(f"State Dim: {state_dim}")
#print(f"Action Dim: {action_dim}")

policy_net = UNOarm_sign_based(state_dim=state_dim, action_dim=action_dim, gridDim=30).to(device)
policy_net.load_state_dict("target_state_dict_test.pth")
target_net = UNOarm_sign_based(state_dim=state_dim, action_dim=action_dim, gridDim=30).to(device)
target_net.load_state_dict(policy_net.state_dict())  # Copy the weights from the policy network to the target network

if __name__ == "__main__":
    # 2.) Training Loop
    episodes = 250
    BATCH_SIZE = 64
    GAMMA = 0.75
    EPSILON_START = 0.9995
    EPSILON_END = 0.01
    EPSILON_DECAY = 30000
    LR = 1.5e-4
    CRITERION = nn.SmoothL1Loss()
    OPTIMIZER = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)
    
    print("Starting Training...")

    policy_network, target_network, reward_sequences = train(
        policy_network=policy_net, target_network=target_net,
        episodes=episodes,
        batch_size=BATCH_SIZE,
        epsilon=EPSILON_START, epsilon_end = EPSILON_END, epsilon_decay=EPSILON_DECAY,
        gamma=GAMMA,
        criterion=CRITERION, optimizer=OPTIMIZER,
        env = JustDoItV1()
    )

    def save_models():
        torch.save(target_network.state_dict(), 'AI/v1_1arm/v3_models/target_state_dict_test.pth')
        torch.save(target_network, 'AI/v1_1arm/v3_models/target_model_test.pth')
        torch.save(policy_network.state_dict(), 'AI/v1_1arm/v3_models/policy_state_dict_test.pth')
        torch.save(policy_network, 'AI/v1_1arm/v3_models/policy_network_test.pth')

    save_models()

    plt.close('all')   # closes any old figures

    ma = moving_average(reward_sequences, window_size=10)

    plt.figure(figsize=(12, 8))

    plt.plot(reward_sequences, label='Episode Reward')

    plt.plot(range(len(ma)), ma, label='10‐Episode Moving Avg')

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Cumulative Reward", fontsize=14)
    plt.title("Training Performance Over Episodes", fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.legend(loc="upper left", fontsize=12)

    plt.tight_layout()
    plt.savefig("AI/v1_1arm/v3_models/Rewards.png", dpi=300)
    plt.show()