import gymnasium as gym
import numpy as np
from climbr import *
import matplotlib.pyplot as plt
import pygame 
class JustDoIt(gym.Env):
    def __init__(self, gridDim = 50, holds = np.column_stack((np.full((100,), 25), np.linspace(0, 50, 100, endpoint=False))), angleChange = 10, energy=500):
        self.gridDim = gridDim
        self.angleChange = angleChange
        target_idx = np.random.randint(0, len(holds))
        self.target_hold = holds[target_idx] # The last possible hold
        self.holds = holds

        self.climbr = climbr() # Let's just use default characteristics
        self.energy = self.climbr.energy
        self.rewards = []

        self.past_distance_deltas_torso = set()
        self.past_distance_deltas_armR = set()
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
        print(self.target_hold[0])
        print(self.target_hold[1])
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
        self.climbr = climbr()
        self.target_hold = self.holds[-1]
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
    
    def step(self, action): # Remember the action is typically sampled stochastically until the policy has been refine by the optimal q values, and then the policy pi will behave greedily
        end = False
        reward = 0

        if action == 0:
            self.climbr.arms[0].grab(self.holds) # CHANGE THIS IN THE FUTURE AS YOU HAVE MORE ARMS MOST LIKELY (LMAOOO)

            arm_location = self.climbr.arms[0].location

            if(arm_location[0] > 50 or arm_location[1] > 50 or arm_location[0] < 0 or arm_location[1] < 0):
                end = True
                reward = -1000

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
            boost = cosine_sim * 3
            if(boost < 0):
                boost /=2
            print(f"Boost: {boost}")
            reward += 1 + boost

        elif action == 1:
            self.climbr.arms[0].release()           

            torso_target_vector = (self.target_hold[0] - self.climbr.torso.location[0],
                                    self.target_hold[1] -self.climbr.torso.location[1])
            # Torso->Arm Vector
            torso_arm_vector = (self.climbr.arms[0].location[0] - self.climbr.torso.location[0],
                                self.climbr.arms[0].location[1] - self.climbr.torso.location[1])
            
            # Cosine Similarity

            cosine_sim = np.dot(torso_target_vector, torso_arm_vector) / (np.linalg.norm(torso_target_vector) * np.linalg.norm(torso_arm_vector))
                # Note: range is (-1, 1)
            boost = cosine_sim * 3
            if(boost < 0):
                boost /=2
            print(f"Boost: {boost}")
            reward += 1 - boost

        elif action == 2:
            self.climbr.shift_arm(0, self.angleChange)
        elif action == 3:
            self.climbr.shift_arm(0, -1 * self.angleChange)

        reward += -0.5

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

        if(np.sign(average_body_delta) < 0):
            reward += -100
        elif(np.sign(average_body_delta) > 0):
            reward += 100
        #reward += 50 * np.sign(average_body_delta) * np.sqrt(torso_distance_delta**2 + arm_distance_delta**2) # -1 is makes it better for the negative delta because that means closer

        # Discourages to be the same distance away 
        # Implement some check backwards 18 elements because at most it should be rotating 180 degrees an arm
        
        torso_loc_tuple = tuple(self.climbr.torso.location)
        arm_loc_tuple = tuple(self.climbr.arms[0].location)
        if(torso_loc_tuple in self.past_distance_deltas_torso and arm_loc_tuple in self.past_distance_deltas_armR):
            if action == 0 or action == 1:
                reward += -1.5
            if action == 2 or action == 3:
                reward += -10
            
                #print("HERE")
        self.past_distance_deltas_torso.add(tuple(self.climbr.torso.location))
        self.past_distance_deltas_armR.add(tuple(self.climbr.arms[0].location))

        #print(f"ABD: {average_body_delta}")
        #print(f"TD: {torso_distance_delta}")
        #print(f"AD: {arm_distance_delta}")
        if(np.linalg.norm(self.target_hold - self.climbr.arms[0].location) <= 1.5): # Within Arm Range
           end = True
           reward += 200

        #else:
        #    if self.energy <= 0:
        #        end = True
        #        reward += -50
        
        # Speed Up Reward logic
        #self.energy += -1

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
        self.inner_state['average_distance_delta'] = average_body_delta
        self.inner_state['distance_from_target_TORSO'] = np.linalg.norm(self.target_hold - self.climbr.torso.location)
        self.inner_state['distance_from_target_ARM'] = np.linalg.norm(self.target_hold - self.climbr.arms[0].location)

        self.rewards.append(reward) #For Future Analysis
        
        print((action, reward), end="|..........|")



        return self.inner_state, reward, end, {}, {}

    def render(self):
        self.ax.clear()

        # Plot holds if available.
        if self.holds.size > 0:
            self.ax.plot(self.holds[:, 0], self.holds[:, 1], 'go', markersize=6, label="Holds")

        # Plot torso location.
        torso_loc = self.climbr.torso.location
        self.ax.plot(torso_loc[0], torso_loc[1], 'bo', markersize=10, label="Torso")

        # Always use the stored arm location as the endpoint.
        endpoint = self.climbr.arms[0].location

        # Draw the arm from torso to endpoint and mark the endpoint.
        self.ax.plot([torso_loc[0], endpoint[0]],
                    [torso_loc[1], endpoint[1]],
                    'r-', lw=2, label="Arm")
        self.ax.plot(endpoint[0], endpoint[1], 'ko', markersize=6)

        # Set display parameters.
        self.ax.set_xlim(0, 50)
        self.ax.set_ylim(0, 50)
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