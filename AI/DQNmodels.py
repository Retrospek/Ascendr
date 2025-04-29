import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs/experiment_1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is basically just a function approximater that takes the form V(s, w), where the weights are learned in the DQN.
# We are using a Q-Learning (deep) because off of my current intuition there's no sense of risk as of yet when it comes to certain actions taken place.
class UNOarm(nn.Module):
    
    def __init__(self, state_dim, action_dim, gridDim):
        # DQN Model
        # -------------------------------------------
        # Args:
        # - state_dim (int): Dimension of flattened state space
        # - action_dim (int): Number of discrete actions
        # - gridDim (int): Width/height of square image grid (e.g., 50)
        # -------------------------------------------
        super(UNOarm, self).__init__()

        self.gridDim = gridDim

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.convfc1 = nn.Linear(800, 512)
        self.convfc2 = nn.Linear(512, 256)
        self.convfc3 = nn.Linear(256, 128)
        self.convfc4_solo = nn.Linear(128, action_dim)

        self.fc1 = nn.Linear(state_dim - gridDim ** 2, 256) # should be the start but for the sake of running imma hardcode
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
        #combined_output = torch.cat((conv, lin), dim=1)
        #output = self.combined_pred(combined_output)
        return output


class UNOarm_sign_based(nn.Module):

    def __init__(self, state_dim, action_dim, gridDim):
        # DQN Model
        # -------------------------------------------
        # Args:
        # - state_dim (int): Dimension of flattened state space
        # - action_dim (int): Number of discrete actions
        # - gridDim (int): Width/height of square image grid (e.g., 50)
        # -------------------------------------------
        super(UNOarm_sign_based, self).__init__()

        self.gridDim = gridDim

        self.conv1 = nn.Conv2d(1, 16, kernel_size=6)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.convfc1 = nn.Linear(800, 512)
        self.convfc2 = nn.Linear(512, 256)
        self.convfc3 = nn.Linear(256, 128)
        self.convfc4_solo = nn.Linear(128, action_dim)

        self.relu = nn.ReLU()

    def forward(self, input):
        batch_size = input.size(0)

        image = input[:, :self.gridDim * self.gridDim].view(batch_size, 1, self.gridDim, self.gridDim)

        conv = self.pool(self.conv1(image))
        conv = self.pool2(self.conv2(conv))
        conv = torch.flatten(conv, start_dim=1)
        conv = self.convfc1(conv)
        conv = self.convfc2(conv)
        conv = self.convfc3(conv)

        output = self.convfc4_solo(conv) # Leave as a linear output otherwise you're breaking the Bellman Equation Assumption that the q values are non-scaled
    
        return output