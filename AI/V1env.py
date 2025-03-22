import gymnasium as gym
import numpy as np
from climbr import *


class lavaFloor(gym.Env):
    def __init__(self, gridDim = 50, holds = np.array([]), angleChange = math.radians(8)):
        self.gridDim = gridDim

        self.angleChange = angleChange

        target_idx = np.random.randint(0, len(holds) + 1)
        self.target_hold = holds[target_idx] # The last possible hold
        
        self.holds = holds

        self.climbr = climbr() # Let's just use default characteristics

        #CHANGING THIS \/ in the future
        self.action_space = gym.spaces.Discrete(3) # 0: Hold, 1:Let Go, 2:Shifting(Implicitely this takes care of the 2 states of the arm grabbing or not holding innately)
        # In reality it is 4 actions ^^^^^^, but the other two actions are built into one function

        self.observation_space = gym.spaces.Dict(
            {
                "agent": self.climbr.torso.location,
                "holds": self.holds,
                "target_hold": self.target_hold,
                "distance_from_target_TORSO": self.target_hold - self.climbr.torso.location,
                "distance_from_target_ARM": self.target_hold - self.climber.arms[0].location # We are experimenting with one arm, so we'll edit this later lmaoo
            }
        )
    
    def reset(self):
        # Now I need to reset the environment from above
        self.climbr = climbr()
        


    def step(self, action): # Remember the action is typically sampled stochastically until the policy has been refine by the optimal q values, and then the policy pi will behave greedily
        end = False
        if action == 0:
            self.climbr.arms[0].grab(self.holds) # CHANGE THIS IN THE FUTURE AS YOU HAVE MORE ARMS MOST LIKELY (LMAOOO)

            arm_location = self.climbr.arms[0].location

            if(arm_location[0] > 25 or arm_location[1] > 25 or arm_location[0] < -25 or arm_location[1] < -25):
                end = True
        if action == 1:
            self.climbr.arms[0].release()
        if action == 2:
            self.climbr.shift_arm(self, 0, self.angleChange)
        

        """
        Now we have to make sure nothing is breaking
        """








