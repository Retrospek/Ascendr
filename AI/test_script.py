import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from V1env import *  # Ensure this import is correct

if __name__ == "__main__":
    # Create 50 random holds within the -25 to 25 grid
    holds = np.random.randint(low=-25, high=25, size=(50, 2))
    env = JustDoIt(gridDim=50, holds=holds)
    state = env.reset()
    rewards = []

    for _ in range(100):
        action = env.action_space.sample()
        state, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        env.render()
        if done:
            print("Episode finished with reward", reward)
            break

    # Plot rewards after episode finishes
    plt.ioff()  # Turn off interactive mode for final plot
    plt.figure()
    plt.plot(rewards, marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("Rewards over Time")
    plt.grid(True)
    plt.show()

    env.close()
