import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from V1env import JustDoIt 
from DQNagent import *


if __name__ == "__main__":

    target = DQN()
    target.load_state_dict(torch.load('target_state_dict.pth'))
    target.eval()

    holds = np.column_stack((np.full((100,), 2), np.linspace(-25, 25, 100)))

    env = JustDoIt()
    state = env.reset()
    accum_reward = 0
    rewards = []

    done = False
    for _ in range(200):
        with torch.no_grad():
            action = torch.argmax(target(state))

        state, reward, done, info, _ = env.step(action)
        accum_reward += reward
        rewards.append(accum_reward)

        env.render()

        if done:
            print("Episode finished with reward:", reward)
            break

    plt.ioff()  
    plt.figure()
    plt.plot(rewards, marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("Rewards over Time")
    plt.grid(True)
    plt.show()

    env.close()
