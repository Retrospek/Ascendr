import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class arm():
    def __init__(self, torso_loc = np.array([0, 0]), length = 2, angle = 0, grabbing=True):
        self.length = length
        self.angle = np.radians(angle)
        self.location = np.array([torso_loc[0] + length * np.cos(self.angle), torso_loc[1] + length * np.sin(self.angle)])
        self.grabbing = grabbing

    def grab(self, holds): # Need to make sure that the hand is close to a hold or "enough"
        for hold in holds:
            if np.linalg.norm(self.location - hold) < 2:
                self.grabbing = True

    def release(self):
        self.grabbing = False

class torso():
    def __init__(self, location = np.asarray((0, 0))):
        self.location = location

class climbr:
    def __init__(self, start_loc = np.asarray((0, 0)), arm_lengths = np.array([2]), arm_degrees = np.array([0])):
        self.torso = torso(location = start_loc)
        self.arms = [arm(length=arm_lengths[i], angle=arm_degrees[i]) for i in range(len(arm_lengths))] # 0 = Right and 1 = Left <= Indices

    def grab(self, side, holds):
        self.arms[side].grab(holds)

    def release(self, side):
        self.arms[side].release()

    def shift_arm(self, side, angle):
        theta = np.radians(angle)
        if(self.arms[side].grabbing == False):
            self.arms[side].angle += theta
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
            


# --- Create a climbr instance (with one arm) ---
climber_obj = climbr(start_loc=np.array([0, 0]),
                     arm_lengths=np.array([2]),
                     arm_degrees=np.array([45]))

# --- Generate 40 integer (x,y) hold positions ---
num_samples = 40
int_coords = np.random.randint(low=-10, high=11, size=(num_samples, 2))

def render_climbr(ax, climber, holds):
    ax.clear()
    torso_loc = climber.torso.location

    # Plot holds (green circles)
    ax.plot(holds[:, 0], holds[:, 1], 'go', markersize=6, label="Holds")

    # Plot torso
    ax.plot(torso_loc[0], torso_loc[1], 'bo', markersize=10, label="Torso")

    # Compute or use the arm endpoint
    if climber.arms[0].grabbing:
        endpoint = climber.arms[0].location
    else:
        theta = climber.arms[0].angle
        endpoint = torso_loc + climber.arms[0].length * np.array([np.cos(theta), np.sin(theta)])

    # Plot the arm line and endpoint
    ax.plot([torso_loc[0], endpoint[0]], [torso_loc[1], endpoint[1]],
            'r-', lw=2, label="Arm")
    ax.plot(endpoint[0], endpoint[1], 'ko', markersize=6)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Climber State (1 Arm)")
    ax.grid(True)
    ax.legend()
    plt.draw()


# --- Set Up the Figure and Widgets ---
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)

# Render once at the start
render_climbr(ax, climber_obj, int_coords)

ax_angle = plt.axes([0.25, 0.25, 0.65, 0.03])
angle_slider = Slider(
    ax=ax_angle,
    label='Angle Change',
    valmin=-45,
    valmax=45,
    valinit=0,
)

ax_button = plt.axes([0.25, 0.15, 0.15, 0.04])
grab_button = Button(ax_button, 'Toggle Grab')

def update(val):
    angle_change = angle_slider.val
    climber_obj.shift_arm(0, angle_change)
    render_climbr(ax, climber_obj, int_coords)
    angle_slider.reset()

angle_slider.on_changed(update)

def toggle_grab(event):
    if climber_obj.arms[0].grabbing:
        climber_obj.release(0)
        print("Arm released")
    else:
        # Lock the arm endpoint
        torso_loc = climber_obj.torso.location
        theta = climber_obj.arms[0].angle
        locked_endpoint = torso_loc + climber_obj.arms[0].length * np.array([np.cos(theta), np.sin(theta)])
        climber_obj.arms[0].location = locked_endpoint
        climber_obj.grab(0, int_coords)
        print("Arm grabbed")
    render_climbr(ax, climber_obj, int_coords)

grab_button.on_clicked(toggle_grab)

plt.show()