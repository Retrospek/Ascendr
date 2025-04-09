import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from gymnasium.spaces.utils import flatten
from V1env import JustDoIt

if __name__ == "__main__":
    env = JustDoIt()

    obs, _ = env.reset()
    state = flatten(env.observation_space, obs)  
    accum_reward = 0
    rewards = []
    done = False

    for _ in range(200):
        env.render()  # Render the environment
        print("Current cumulative reward:", accum_reward)
        valid_input = False

        while not valid_input:
            try:
                action = int(input("Enter your action (0, 1, 2, or 3): "))
                if action in [0, 1, 2, 3]:
                    valid_input = True
                else:
                    print("Invalid input. Please enter one of 0, 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

        obs, reward, done, info, _ = env.step(action)
        print(f"Image Observation: {obs["environment_image"]}")
        accum_reward += reward
        rewards.append(accum_reward)

        state = flatten(env.observation_space, obs)

        if done:
            print("Episode finished with reward:", accum_reward)
            break

    plt.ioff()
    plt.figure()
    plt.plot(rewards, marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title("Rewards over Time")
    plt.grid(True)
    plt.show()

    env.close()
