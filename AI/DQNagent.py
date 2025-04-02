import numpy as np
import math
import random
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt

from V1env import JustDoIt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# This is basically just a function approximater that has takes the form V(s, w), where the weights are learned in the DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()

        #self.conv1 = nn.Conv2d()

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
        out = self.relu(out) # This part is actually pretty weird, but the policy is what generates the q values for every possible action and then we sample using softmax

        return out

def train(trained_network, target_network, episodes, time_steps, epsilon, epsilon_decay, gamma, gamma_decay, learning_rate):

    all_reward_sequences = []  # For graphing down the line

    for episode in range(episodes):
        accumulated_reward = 0
        episode_reward = []  # Start an empty list for episode rewards
        state = env.reset()
        
        for time_step in range(time_steps):  # Keep running until allocated steps are gone
            
            # TODO: Implement epsilon-greedy action selection using the trained_network
            # - If a random number < epsilon, choose a random action (exploration)
            # - Otherwise, choose the action that maximizes the Q-value from trained_network (exploitation)

            action_epsilon_chance = np.random.rand()

            if(action_epsilon_chance < epsilon):
                sampled_action = env.action_space.sample()  # Placeholder for action selection

                inner_state, reward, end, _, _ = env.step(sampled_action)

                # TODO: Store the experience tuple (state, sampled_action, reward, inner_state, end) into the replay buffer
                
                

            # TODO: If the replay buffer has enough samples, sample a minibatch from it for training

            # TODO: For each sample in the minibatch, compute the target Q-value
            # - Use target_network for stable targets: target = reward + (gamma * max_a' Q_target(inner_state, a') if not end else reward)
            
            # TODO: Compute the predicted Q-values from trained_network for the actions taken

            # TODO: Compute the loss between the predicted Q-values and the target Q-values

            # TODO: Perform backpropagation and update the parameters of the trained_network using the optimizer
            
            # TODO: Optionally update the target_network periodically using the trained_network parameters

            accumulated_reward += reward
            episode_reward.append(accumulated_reward)
            
            # Update state to inner_state for next time step
            state = inner_state

            if end:
                break

        # Optionally, decay epsilon and gamma after each episode for exploration and discounting respectively
        # TODO: Implement epsilon and gamma decay logic

        all_reward_sequences.append(episode_reward)
        
    return trained_network, target_network, all_reward_sequences


""" 
Quick Review
<><><><><><><><><><><><><><>

This is a DQN model, and because of this we know we're trying to approximate the policy function instead of discretizing each
action and state to find some optimal policy and just converging to some optimal q table

Steps:
1. Initialize the env => training network, and the copied over target network
2. Initialize some episodic training loop
    - Grab rewards iterate the states and actions(sampled, with some epsilon factor (decay as well)) blah blah blah
3. Utilize an R+1 expectation state/action value methodology not the very end (n_step = 1 Temporal Difference Learning or MC idk just something with one step)
4. Apply the training of the policy to the network
5. Test that shit out lmfaooo
<><><><><><><><><><><><><><>
"""

# 1.) Setting Up the Environment & The dual Weilding DQN networks

env = JustDoIt()
action_dim = len(env.action_spaces)
state_dim = len(env.observation_spaces) # - Technically a lie because there are variable holds

policy_net = DQN(state_dim = state_dim, action_dim = action_dim)
target_net = DQN(state_dim = state_dim, action_dim = action_dim)

target_net.load_state_dict(policy_net.state_dict()) # Copy the weights over from the changing policy net to the target network

# 2.) Training_loop

episodes = 3000
time_steps = 1000
EPSILON = 0.15
EPSILON_DECAY = 0.98
GAMMA = 0.75
GAMMA_DECAY = 0.95
LR = 0.001


trained_network, target_network, all_reward_sequences = train(trained_network = policy_net, target_network = target_net,
                                                              episodes = episodes, time_steps = time_steps,
                                                              epsilon = EPSILON, epsilon_decay = EPSILON_DECAY,
                                                              gamma = GAMMA, gamma_decay = GAMMA_DECAY,
                                                              learning_rate = LR)
