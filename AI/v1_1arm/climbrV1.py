import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class armV1():
    def __init__(self, torso_loc = np.array([15, 15]), length = 2, angle = 0, grabbing=True):
        self.length = length
        self.angle = np.radians(angle)
        self.location = np.array([torso_loc[0] + length * np.cos(self.angle), torso_loc[1] + length * np.sin(self.angle)])
        self.grabbing = grabbing

    def grab(self, holds): # Need to make sure that the hand is close to a hold or "enough"
        for hold in holds:
            if np.linalg.norm(self.location - hold) < 0.75:
                self.grabbing = True

    def release(self):
        self.grabbing = False

class torso():
    def __init__(self, location = np.asarray((24, 24))):
        self.location = location

class climbrV1:
    def __init__(self, start_loc = np.asarray((15, 15)), arm_lengths = np.array([2]), arm_degrees = np.array([10]), energy=750):
        self.torso = torso(location = start_loc)
        self.arms = [armV1(length=arm_lengths[i], angle=arm_degrees[i], torso_loc=self.torso.location) for i in range(len(arm_lengths))] # 0 = Right and 1 = Left <= Indices
        self.energy = energy
    def grab(self, side, holds):
        self.arms[side].grab(holds)

    def release(self, side):
        self.arms[side].release()
    
    def shift_arm(self, side, angle):
        theta = np.radians(angle)
        if(self.arms[side].grabbing == False):
            self.arms[side].angle += theta
            arm_angle = self.arms[side].angle
            self.arms[side].location = np.array([self.torso.location[0] + self.arms[side].length * np.cos(arm_angle), self.torso.location[1] + self.arms[side].length * np.sin(arm_angle)])
        else: # If we are in fact grabbing we want to make 
            theta *= -1
            # Now we need to shift every point around the limb that's rotating itself kind of when grabbing
            # Must use initial torso position to find the rotating limb's initial position

            self.arms[side].angle += theta # Because I'm not tryna reset the arm position, but just leave it wherever it was, so update theta value
            
            rotation_matrix = np.array([np.cos(theta), -1 * np.sin(theta), np.sin(theta), np.cos(theta)]).reshape(2, 2)

            arm_location = self.arms[side].location # Will use this as a pivot for the torso

            normalized_torso_position = np.asarray((self.torso.location[0] - arm_location[0], self.torso.location[1] - arm_location[1]))

            normalized_rotate_body_coords = np.dot(rotation_matrix, normalized_torso_position)

            # A3 = A2 + C
            
            self.torso.location = normalized_rotate_body_coords + arm_location