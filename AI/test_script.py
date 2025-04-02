import numpy as np
import matplotlib.pyplot as plt
from V1env import JustDoIt 

if __name__ == "__main__":
    holds = np.column_stack((np.full((100,), 2), np.linspace(-25, 25, 100)))

    env = JustDoIt()
    state = env.reset()
    accum_reward = 0
    rewards = []

    done = False
    for _ in range(200):
        action = env.action_space.sample()

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
