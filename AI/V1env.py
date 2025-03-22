import gymnasium as gym
import numpy as np
from climbr import *
import matplotlib.pyplot as plt

class lavaFloor(gym.Env):
    def __init__(self, gridDim = 50, holds = np.array([]), angleChange = math.radians(8)):
        self.gridDim = gridDim
        self.angleChange = angleChange
        target_idx = np.random.randint(0, len(holds) + 1)
        self.target_hold = holds[target_idx] # The last possible hold
        self.holds = holds

        self.climbr = climbr() # Let's just use default characteristics
        self.rewards = []
        #CHANGING THIS \/ in the future
        self.action_space = gym.spaces.Discrete(3) # 0: Hold, 1:Let Go, 2:Shifting(Implicitely this takes care of the 2 states of the arm grabbing or not holding innately)
        # In reality it is 4 actions ^^^^^^, but the other two actions are built into one function
    
        self.observation_space = gym.spaces.Dict(
            {
                "agent": self.climbr.torso.location,
                "arm_location": self.climbr.arms[0].location,
                "holds": self.holds,
                "target_hold": self.target_hold,
                "distance_from_target_TORSO": np.linalg.norm(self.target_hold - self.climbr.torso.location),
                "distance_from_target_ARM": np.linalg.norm(self.target_hold - self.climber.arms[0].location) # We are experimenting with one arm, so we'll edit this later lmaoo

            }
        )

        self.inner_state = {
            "torso_location": self.climbr.torso.location,
            "arm_location": self.climbr.arms[0].location,
            "holds": self.holds,
            "target_hold": self.target_hold,
            "distance_from_target_TORSO": np.linalg.norm(self.target_hold - self.climbr.torso.location),
            "distance_from_target_ARM": np.linalg.norm(self.target_hold - self.climber.arms[0].location)
        }
    
    def reset(self):
        # Now I need to reset the environment from above
        self.climbr = climbr()
        self.target_hold = self.holds[np.random.randint(0, len(self.holds))]
        self.rewards = []

        self.observation_space = gym.spaces.Dict(
            {
                "agent": self.climbr.torso.location,
                "arm_location": self.climbr.arms[0].location,
                "holds": self.holds,
                "target_hold": self.target_hold,
                "distance_from_target_TORSO": np.linalg.norm(self.target_hold - self.climbr.torso.location),
                "distance_from_target_ARM": np.linalg.norm(self.target_hold - self.climber.arms[0].location) # We are experimenting with one arm, so we'll edit this later lmaoo

            }
        )

        self.inner_state = {
            "torso_location": self.climbr.torso.location,
            "arm_location": self.climbr.arms[0].location,
            "holds": self.holds,
            "target_hold": self.target_hold,
            "distance_from_target_TORSO": np.linalg.norm(self.target_hold - self.climbr.torso.location),
            "distance_from_target_ARM": np.linalg.norm(self.target_hold - self.climber.arms[0].location)
        }

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
            self.climbr.shift_arm(self, 0, self.angleChange)
        
        if self.climbr.torso.location == self.target_hold:
            end = True
            reward += 100

        original_distance_from_target_TORSO = self.inner_state['distance_from_target_TORSO']
        new_distance_from_target_TORSO = np.linalg.norm(self.target_hold - self.climbr.torso.location)
        reward += (original_distance_from_target_TORSO - new_distance_from_target_TORSO)

        original_distance_from_target_ARM = self.inner_state['distance_from_target_ARM']
        new_distance_from_target_ARM = np.linalg.norm(self.target_hold - self.climbr.arms[0].location)
        reward += (original_distance_from_target_ARM - new_distance_from_target_ARM)

        if(np.linalg.norm(self.target_hold - self.climbr.arms[0].location) <= self.climbr.arms[0].length): # Close to completion reward
           reward += 10

        reward += -1 # Incorporate some speed factor

        self.inner_state['torso_location'] = self.climbr.torso.location
        self.inner_state['arm_location'] = self.climbr.arms[0].location
        self.inner_state['distance_from_target_TORSO'] = np.linalg.norm(self.target_hold - self.climbr.torso.location)
        self.inner_state['distance_from_target_ARM'] = np.linalg.norm(self.target_hold - self.climber.arms[0].location)

        self.rewards.append(reward) #For Future Analysis

        return self.inner_state, reward, end

    def render(self):
        
        # Turn on interactive mode for low-latency updates
        plt.ion()
        
        # Create a new figure and axis (or reuse an existing one if you prefer)
        fig, ax = plt.subplots()
        ax.clear()
        
        # Get the current torso location from the climbr object
        torso_loc = self.climbr.torso.location
        
        # Ensure that holds is a NumPy array (in case it was provided as a list)
        holds = self.holds if isinstance(self.holds, np.ndarray) else np.array(self.holds)
        
        # Plot holds as green circles (if there are any holds)
        if holds.size > 0:
            ax.plot(holds[:, 0], holds[:, 1], 'go', markersize=6, label="Holds")
        
        # Plot the torso as a blue circle
        ax.plot(torso_loc[0], torso_loc[1], 'bo', markersize=10, label="Torso")
        
        # Compute the arm endpoint: if grabbing, use the current arm location; 
        # otherwise, compute it from the arm's angle and length relative to the torso.
        if self.climbr.arms[0].grabbing:
            endpoint = self.climbr.arms[0].location
        else:
            theta = self.climbr.arms[0].angle
            endpoint = torso_loc + self.climbr.arms[0].length * np.array([np.cos(theta), np.sin(theta)])
        
        # Draw the arm as a red line from the torso to the endpoint, and plot the endpoint as a black dot
        ax.plot([torso_loc[0], endpoint[0]], [torso_loc[1], endpoint[1]], 'r-', lw=2, label="Arm")
        ax.plot(endpoint[0], endpoint[1], 'ko', markersize=6)
        
        # Set axis limits and labels
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Climber State (1 Arm)")
        ax.grid(True)
        ax.legend()
        
        # Draw and pause briefly to update the plot
        plt.draw()
        plt.pause(0.001)








