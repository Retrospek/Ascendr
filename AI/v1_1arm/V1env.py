import gymnasium as gym
import numpy as np
from v1_1arm.climbrV1 import climbrV1
import math
import matplotlib.pyplot as plt
class JustDoItV1(gym.Env):
    def __init__(self, gridDim = 30, holds = np.unique(np.rint(np.column_stack((15 + 1.5*np.sin(np.linspace(0, 4*np.pi, 100)), np.linspace(0, 29, 100, endpoint=False)))).astype(int), axis=0),
                  angleChange = 10, energy=500):
        self.gridDim = gridDim
        self.angleChange = angleChange
        self.holds = holds

        max_y = np.max(self.holds[:,1])
        candidates = self.holds[self.holds[:,1] == max_y]

        # from those, pick the one closest to x=15
        idx = np.argmin(np.abs(candidates[:,0] - 15))
        self.target_hold = candidates[idx]

        self.climbr = climbrV1() # Let's just use default characteristics
        self.energy = self.climbr.energy
        self.rewards = []

        self.past_action = -1
        self.past_positions = set()

        #CHANGING THIS \/ in the future
        self.action_space = gym.spaces.Discrete(4) # 0: Hold, 1:Let Go, 2:Shifting(Implicitely this takes care of the 2 states of the arm grabbing or not holding innately)
        # In reality it is 4 actions ^^^^^^, but the other two actions are built into one function
        # The Observation Space Structure
        self.observation_space = gym.spaces.Dict(
            {
                "environment_image": gym.spaces.Box(low=0, high = 4, shape=(gridDim, gridDim), dtype=np.int32),
                "torso_location": gym.spaces.Box(low=-25, high=25, shape=(2,), dtype=np.float32),
                "arm_location": gym.spaces.Box(low=-25, high=25, shape=(2,), dtype=np.float32),
                "arm_grabbing_status": gym.spaces.MultiBinary(len(self.climbr.arms)),
                #"holds": gym.spaces.Box(low=-25, high=25, shape=(len(self.holds), 2), dtype=np.float32),
                "average_distance_delta": gym.spaces.Box(low=-25, high=25, shape=(1,), dtype=np.float32),
                "target_hold": gym.spaces.Box(low=-25, high=25, shape=(2,), dtype=np.float32),
                "distance_from_target_TORSO": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "distance_from_target_ARM": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)  # We are experimenting with one arm, so we'll edit this later
            }
        )

        # The Actual Observation Space

        environment_picture = np.zeros(shape=(gridDim,gridDim), dtype=np.int32)
        
        for hold in self.holds:
            environment_picture[math.floor(hold[0])][math.floor(hold[1])] = 2
        for arm in self.climbr.arms:
            arm_x = min(max(0, math.floor(arm.location[0])), self.gridDim - 1)
            arm_y = min(max(0, math.floor(arm.location[1])), self.gridDim - 1)
            environment_picture[arm_x, arm_y] = 3
        environment_picture[math.floor(self.target_hold[0])][math.floor(self.target_hold[1])] = 1

        torso_x = min(max(0, math.floor(self.climbr.torso.location[0])), self.gridDim - 1)
        torso_y = min(max(0, math.floor(self.climbr.torso.location[1])), self.gridDim - 1)
        environment_picture[torso_x][torso_y] = 4
        environment_picture = environment_picture.transpose()
        self.environment_picture = environment_picture

        self.inner_state = {
            "environment_image": self.environment_picture,
            "torso_location": self.climbr.torso.location,
            "arm_location": self.climbr.arms[0].location,
            "arm_grabbing_status": np.array([self.climbr.arms[i].grabbing for i in range(len(self.climbr.arms))], dtype=np.int8),
            #"holds": self.holds,
            "average_distance_delta": 24.99,
            "target_hold": self.target_hold,
            "distance_from_target_TORSO": np.linalg.norm(self.target_hold - self.climbr.torso.location),
            "distance_from_target_ARM": np.linalg.norm(self.target_hold - self.climbr.arms[0].location)
        }

        self.fig, self.ax = plt.subplots()
        

    def update_goal_radius(self, current_episode, max_episodes):
        initial_radius = 3.0
        final_radius = 1.5
        decay = current_episode / max_episodes
        self.goal_radius = initial_radius * (1 - decay) + final_radius * decay

    def reset(self):
        # Now I need to reset the environment from above
        self.climbr = climbrV1()

        self.past_action = -1
        self.past_positions = set()

        self.rewards = []
        self.energy = self.climbr.energy
        
        environment_picture = np.zeros(shape=(self.gridDim,self.gridDim), dtype=np.int32)
        for hold in self.holds:
            environment_picture[math.floor(hold[0])][math.floor(hold[1])] = 2
        for arm in self.climbr.arms:
            arm_x = min(max(0, math.floor(arm.location[0])), self.gridDim - 1)
            arm_y = min(max(0, math.floor(arm.location[1])), self.gridDim - 1)
            environment_picture[arm_x, arm_y] = 3
        environment_picture[math.floor(self.target_hold[0])][math.floor(self.target_hold[1])] = 1

        torso_x = min(max(0, math.floor(self.climbr.torso.location[0])), self.gridDim - 1)
        torso_y = min(max(0, math.floor(self.climbr.torso.location[1])), self.gridDim - 1)
        environment_picture[torso_x][torso_y] = 4

        environment_picture = environment_picture.transpose()
        self.environment_picture = environment_picture

        self.inner_state = {
            "environment_image": self.environment_picture,
            "torso_location": self.climbr.torso.location,
            "arm_location": self.climbr.arms[0].location,
            "arm_grabbing_status": np.array([self.climbr.arms[i].grabbing for i in range(len(self.climbr.arms))], dtype=np.int8),
            #"holds": self.holds,
            "average_distance_delta": 24.99,
            "target_hold": self.target_hold,
            "distance_from_target_TORSO": np.linalg.norm(self.target_hold - self.climbr.torso.location),
            "distance_from_target_ARM": np.linalg.norm(self.target_hold - self.climbr.arms[0].location)
        }
        return self.inner_state, {}
    
    def step(self, action): # Remember the action is typically sampled on the concept of correct "approximations" until the policy has been refine by the optimal q values, and then the policy pi will behave greedily
        end = False
        reward = 0

        if action in (0, 1):
            self.climbr.arms[0].grab(self.holds) if action == 0 else self.climbr.arms[0].release()# CHANGE THIS IN THE FUTURE AS YOU HAVE MORE ARMS MOST LIKELY (LMAOOO)

            arm_location = self.climbr.arms[0].location

            #if(arm_location[0] > self.gridDim or arm_location[1] > self.gridDim or arm_location[0] < 0 or arm_location[1] < 0):
                #end = True
                #reward = -500.0   

            # Theta/Vector direction logic
            # Torso->Target Vector
            torso_target_vector = (self.target_hold[0] - self.climbr.torso.location[0],
                                    self.target_hold[1] -self.climbr.torso.location[1])
            # Torso->Arm Vector
            torso_arm_vector = (self.climbr.arms[0].location[0] - self.climbr.torso.location[0],
                                self.climbr.arms[0].location[1] - self.climbr.torso.location[1])
            
            # Cosine Similarity

            cosine_sim = np.dot(torso_target_vector, torso_arm_vector) / (np.linalg.norm(torso_target_vector) * np.linalg.norm(torso_arm_vector))
                # Note: range is (-1, 1)
            
            boost_scale = 1.0
            #print(f"BOOST: {boost_scale * max(0.0, cosine_sim) if action == 0 else boost_scale * max(0.0, -cosine_sim)}")
            reward += boost_scale * max(0.0, cosine_sim) if action == 0 else boost_scale * max(0.0, -cosine_sim)

        elif action in (2, 3):
            self.climbr.shift_arm(0, self.angleChange) if action ==2  else self.climbr.shift_arm(0, -1 * self.angleChange)

        reward += -1

        #Incorporating a average limb and torso accumalated difference to target
        original_distance_from_target_TORSO = self.inner_state['distance_from_target_TORSO']
        new_distance_from_target_TORSO = np.linalg.norm(self.target_hold - self.climbr.torso.location)
        #print(f"TH: {self.target_hold}")
        #print(f"STL: {self.climbr.torso.location}")
        torso_distance_delta = original_distance_from_target_TORSO - new_distance_from_target_TORSO


        original_distance_from_target_ARM = self.inner_state['distance_from_target_ARM']
        new_distance_from_target_ARM = np.linalg.norm(self.target_hold - self.climbr.arms[0].location)
        arm_distance_delta = original_distance_from_target_ARM - new_distance_from_target_ARM

        average_body_delta = (torso_distance_delta + arm_distance_delta) / 2.0

        # Directional Reward
        #print(f"Average Body Delta: {average_body_delta}")
        reward += 40 * average_body_delta

        #reward += 50 * np.sign(average_body_delta) * np.sqrt(torso_distance_delta**2 + arm_distance_delta**2) # -1 is makes it better for the negative delta because that means closer
        
        torso_loc_tuple = tuple(self.climbr.torso.location)
        arm_loc_tuple = tuple(self.climbr.arms[0].location)

        if((torso_loc_tuple, arm_loc_tuple) in self.past_positions):
            if action in (2, 3):
                #print("Caught 2 or 3")
                reward += -5 # THIS SHOULD NEVER HAPPEN LIKE IT'S SO DUMB TO MOVE BACK TO A PLACE YOU WERE AT BEFORE
            else:
                #print("Caught 0 or 1")
                reward += -0.25
        if action == self.past_action and action in (0, 1) or (action in (2, 3) and self.past_action in (2, 3) and action != self.past_action):
            #print("Previous position caught")
            reward += -2

        self.past_action = action
        
        self.past_positions.add((torso_loc_tuple, arm_loc_tuple))

        #print(f"ABD: {average_body_delta}")
        #print(f"TD: {torso_distance_delta}")
        #print(f"AD: {arm_distance_delta}")
        if(np.linalg.norm(self.target_hold - self.climbr.arms[0].location) <= 1.5 or np.linalg.norm(self.target_hold - self.climbr.torso.location) <= 1.5): # Within Arm Range
           end = True
           print("Goal Reached")
           reward += 500.0   

        environment_picture = np.zeros((self.gridDim, self.gridDim), dtype=np.int32)
        for hold in self.holds:
            environment_picture[math.floor(hold[0]), math.floor(hold[1])] = 2
        for arm in self.climbr.arms:
            arm_x = min(max(0, math.floor(arm.location[0])), self.gridDim - 1)
            arm_y = min(max(0, math.floor(arm.location[1])), self.gridDim - 1)
            environment_picture[arm_x, arm_y] = 3
        environment_picture[math.floor(self.target_hold[0])][math.floor(self.target_hold[1])] = 1

        torso_x = min(max(0, math.floor(self.climbr.torso.location[0])), self.gridDim - 1)
        torso_y = min(max(0, math.floor(self.climbr.torso.location[1])), self.gridDim - 1)
        environment_picture[torso_x][torso_y] = 4
        environment_picture = environment_picture.transpose()

        self.environment_picture = environment_picture
        self.inner_state['environment_image'] = self.environment_picture
        self.inner_state['torso_location'] = self.climbr.torso.location
        self.inner_state['arm_location'] = self.climbr.arms[0].location
        self.inner_state["arm_grabbing_status"] = np.array([self.climbr.arms[i].grabbing for i in range(len(self.climbr.arms))], dtype=np.int8),
        self.inner_state['average_distance_delta'] = np.array([average_body_delta], dtype=np.float32)
        self.inner_state['distance_from_target_TORSO'] = np.linalg.norm(self.target_hold - self.climbr.torso.location)
        self.inner_state['distance_from_target_ARM'] = np.linalg.norm(self.target_hold - self.climbr.arms[0].location)

        self.rewards.append(reward) #For Future Analysis
        
        #print(f"#### ACTION: {action} => REWARD: {reward} ####")

        return self.inner_state, reward, end, False, {}

    def render(self, show=True):
        self.ax.clear()

        if self.holds.size > 0:
            self.ax.plot(self.holds[:, 0], self.holds[:, 1], 'go', markersize=6, label="Holds")

        torso_loc = self.climbr.torso.location
        self.ax.plot(torso_loc[0], torso_loc[1], 'bo', markersize=10, label="Torso")

        endpoint = self.climbr.arms[0].location
        self.ax.plot([torso_loc[0], endpoint[0]],
                    [torso_loc[1], endpoint[1]],
                    'r-', lw=2, label="Arm")
        self.ax.plot(endpoint[0], endpoint[1], 'ko', markersize=6)

        self.ax.set_xlim(0, self.gridDim)
        self.ax.set_ylim(0, self.gridDim)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Climber State (1 Arm)")
        self.ax.grid(True)
        self.ax.legend()

        self.fig.canvas.draw()
        if show:
            plt.show(block=False)
            plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)