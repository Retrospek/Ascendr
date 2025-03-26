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
from climbr import climbr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
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
        out = self.relu(out) # This part is actually pretty weird, but the policy is what generates probabilities no the softmax equation weirdly enough

        return out
    
EPSILON = 0.15
EPSILON_DECAY = 0.95
GAMMA = 0.75
LR = 0.001

def train(trained_network, target_network, episodes, time_steps, epsilon, epsilon_decay, gamma, gamma_decay):
    for episode in range(episodes):
        for time_step in range(time_steps): # Keep Running until allocated steps are gone
            state = env.reset()

            sampled_action = env.action_space.sample()

            inner_state, reward, end, _, _ = env.step(sampled_action)



""" 
Quick Review
<><><><><><><><><><><><><><>

This is a DQN model, and because of this we know we're trying to approximate the policy function instead of discretizing each
action and state to find some optimal policy and just converging to some optimal q table

Steps:
1. Initialize the env => training network, and the copied over target network
2. Initialize some episodic training loop
    - Grab rewards iterate the states and actions(sampled, with some epsilon factor (decay as well)) blah blah blah
3. Utilize an R+1 expectation state/action value methodology not the very end 
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

target_net.load_state_dict(policy_net.state_dict())

# 2.) Training_loop



