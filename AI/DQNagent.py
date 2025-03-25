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

class DQN(nn.Module):
    def __init__(self, state, actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, actions)

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

""" 
Quick Review
<><><><><><><><><><><><><><>

This is a DQN model, and because of this we know we're trying to approximate the policy function instead of discretizing each
action and state to find some optimal policy and just converging to some optimal q table

Steps:
1. Initialize the training network, and the copied over target network
2. Initialize some episodic training loop
    - Grab rewards iterate the states and actions(sampled, with some epsilon factor (decay as well)) blah blah blah
3. Utilize an R+1 expectation state/action value methodology not the very end 
4. Apply the training of the policy to the network
5. Test that shit out lmfaooo
<><><><><><><><><><><><><><>
"""

