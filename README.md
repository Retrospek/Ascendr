# Ascendr

Ascendr is a climbing simulation environment built on top of [Gymnasium](https://gymnasium.farama.org/). The project provides a (someday lol) physics-based simulation where a climber interacts with holds on a climbing wall. Whether you're experimenting with reinforcement learning algorithms like DQN or just want to play manually, Ascendr offers a versatile platform for climbing simulation.


https://github.com/user-attachments/assets/359016dd-8d1a-49f2-9488-fc55400a9da7



https://github.com/user-attachments/assets/602079c9-ffea-47e6-a16d-e193a47bb91b


## Features

- **Custom Gym Environment**: Implements the `JustDoIt` environment with a discrete action space.
- **Action Space**:
  - `0`: Grab a hold
  - `1`: Release hold
  - `2`: Shift arm (e.g., move left)
  - `3`: Shift arm (e.g., move right)
- **Observation Space**:
  - Climber’s torso and arm positions
  - Available holds on the wall
  - Target hold details
  - Distance metrics to the target
- **Real-time Visualization**: Uses `matplotlib` to render the climber’s state and environment.
- **Reinforcement Learning Ready**: Easily integrate with different algorithm architectures such as DQN for training experiments.

## Installation

### Prerequisites

- Python 3.7 or higher

### Steps

1. **Please FORK this lol**
   git pull blah blah blah
