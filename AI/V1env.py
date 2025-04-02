import gymnasium as gym
import numpy as np
from climbr import *
import matplotlib.pyplot as plt

class JustDoIt(gym.Env):
    def __init__(self, gridDim = 50, holds = np.column_stack((np.full((100,), 2), np.linspace(-25, 25, 100))), angleChange = 10):
        self.gridDim = gridDim
        self.angleChange = angleChange
        target_idx = np.random.randint(0, len(holds) + 1)
        self.target_hold = holds[target_idx] # The last possible hold
        self.holds = holds

        self.climbr = climbr() # Let's just use default characteristics
        self.rewards = []
        #CHANGING THIS \/ in the future
        self.action_space = gym.spaces.Discrete(4) # 0: Hold, 1:Let Go, 2:Shifting(Implicitely this takes care of the 2 states of the arm grabbing or not holding innately)
        # In reality it is 4 actions ^^^^^^, but the other two actions are built into one function
    
        self.observation_space = gym.spaces.Dict(
            {
                "torso_location": gym.spaces.Box(low=-25, high=25, shape=(2,), dtype=np.float32),
                "arm_location": gym.spaces.Box(low=-25, high=25, shape=(2,), dtype=np.float32),
                "holds": gym.spaces.Box(low=-25, high=25, shape=(len(self.holds), 2), dtype=np.float32),
                "target_hold": gym.spaces.Box(low=-25, high=25, shape=(2,), dtype=np.float32),
                "distance_from_target_TORSO": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "distance_from_target_ARM": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)  # We are experimenting with one arm, so we'll edit this later
            }
        )

        self.inner_state = {
            "torso_location": self.climbr.torso.location,
            "arm_location": self.climbr.arms[0].location,
            "holds": self.holds,
            "target_hold": self.target_hold,
            "distance_from_target_TORSO": np.linalg.norm(self.target_hold - self.climbr.torso.location),
            "distance_from_target_ARM": np.linalg.norm(self.target_hold - self.climbr.arms[0].location)
        }

        self.fig, self.ax = plt.subplots()

    def q_table_state_discretizer(self):
        """
        States:
        0 = Below Target
        1 = Above Target
        2 = Holding Target
        3 = Not Holding Target
        """

    def reset(self):
        # Now I need to reset the environment from above
        self.climbr = climbr()
        self.target_hold = self.holds[np.random.randint(0, len(self.holds))]
        self.rewards = []

        self.inner_state = {
            "torso_location": self.climbr.torso.location,
            "arm_location": self.climbr.arms[0].location,
            "holds": self.holds,
            "target_hold": self.target_hold,
            "distance_from_target_TORSO": np.linalg.norm(self.target_hold - self.climbr.torso.location),
            "distance_from_target_ARM": np.linalg.norm(self.target_hold - self.climbr.arms[0].location)
        }
        return self.inner_state, {}
    
    def step(self, action): # Remember the action is typically sampled stochastically until the policy has been refine by the optimal q values, and then the policy pi will behave greedily
        end = False
        reward = 0

        if action == 0:
            self.climbr.arms[0].grab(self.holds) # CHANGE THIS IN THE FUTURE AS YOU HAVE MORE ARMS MOST LIKELY (LMAOOO)

            arm_location = self.climbr.arms[0].location

            if(arm_location[0] > 25 or arm_location[1] > 25 or arm_location[0] < -25 or arm_location[1] < -25):
                end = True
        if action == 1:
            self.climbr.arms[0].release()
        if action == 2:
            self.climbr.shift_arm(0, self.angleChange)
        if action == 3:
            self.climbr.shift_arm(0, -1 * self.angleChange)

        if self.climbr.torso.location[0] == self.target_hold[0] and self.climbr.torso.location[1] == self.target_hold[1]:
            end = True
            reward += 100

        original_distance_from_target_TORSO = self.inner_state['distance_from_target_TORSO']
        new_distance_from_target_TORSO = np.linalg.norm(self.target_hold - self.climbr.torso.location)
        reward += (original_distance_from_target_TORSO - new_distance_from_target_TORSO) * 2

        #original_distance_from_target_ARM = self.inner_state['distance_from_target_ARM']
        #new_distance_from_target_ARM = np.linalg.norm(self.target_hold - self.climbr.arms[0].location)
        #reward += (original_distance_from_target_ARM - new_distance_from_target_ARM) * 2

        if(np.linalg.norm(self.target_hold - self.climbr.arms[0].location) <= self.climbr.arms[0].length): # Close to completion reward
           reward += 10

        reward += -1 # Incorporate some speed factor

        self.inner_state['torso_location'] = self.climbr.torso.location
        self.inner_state['arm_location'] = self.climbr.arms[0].location
        self.inner_state['distance_from_target_TORSO'] = np.linalg.norm(self.target_hold - self.climbr.torso.location)
        self.inner_state['distance_from_target_ARM'] = np.linalg.norm(self.target_hold - self.climbr.arms[0].location)

        self.rewards.append(reward) #For Future Analysis

        return self.inner_state, reward, end, {}, {}

    def render(self):
        self.ax.clear()

        if self.holds.size > 0:
            self.ax.plot(self.holds[:, 0], self.holds[:, 1], 'go', markersize=6, label="Holds")

        torso_loc = self.climbr.torso.location
        self.ax.plot(torso_loc[0], torso_loc[1], 'bo', markersize=10, label="Torso")

        if self.climbr.arms[0].grabbing:
            endpoint = self.climbr.arms[0].location
        else:
            theta = self.climbr.arms[0].angle
            endpoint = torso_loc + self.climbr.arms[0].length * np.array([np.cos(theta), np.sin(theta)])
        self.ax.plot([torso_loc[0], endpoint[0]],
                    [torso_loc[1], endpoint[1]],
                    'r-', lw=2, label="Arm")
        self.ax.plot(endpoint[0], endpoint[1], 'ko', markersize=6)

        self.ax.set_xlim(-25, 25)
        self.ax.set_ylim(-25, 25)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Climber State (1 Arm)")
        self.ax.grid(True)
        self.ax.legend()

        self.fig.canvas.draw()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)