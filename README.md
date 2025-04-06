# Ascendr

Ascendr is a climbing simulation environment built on top of [Gymnasium](https://gymnasium.farama.org/). The project provides a physics-based simulation where a climber interacts with holds on a climbing wall. Whether you're experimenting with reinforcement learning algorithms like DQN or just want to play manually, Ascendr offers a versatile platform for climbing simulation.

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
- **Reinforcement Learning Ready**: Easily integrate with RL agents such as DQN for training experiments.

## Installation

### Prerequisites

- Python 3.7 or higher

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ascendr.git
   cd ascendr
   ```
