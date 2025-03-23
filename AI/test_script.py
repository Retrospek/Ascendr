import numpy as np
import matplotlib.pyplot as plt
from V1env import JustDoIt  # Adjust import to match your file/module structure

if __name__ == "__main__":
    # Create 50 random holds within the -25 to 25 grid
    holds = np.column_stack((np.full((100,), 2), np.linspace(-25, 25, 100)))

    # Initialize environment
    env = JustDoIt()
    state = env.reset()
    rewards = []

    # Run for 100 steps
    done = False
    for _ in range(200):
        # Take random action
        action = env.action_space.sample()

        # Step environment
        state, reward, done, info, _ = env.step(action)
        rewards.append(reward)

        # Render environment
        env.render()

        # End loop if done
        if done:
            print("Episode finished with reward:", reward)
            break

    # Plot rewards
    plt.ioff()  # Turn off interactive mode before final plot
    plt.figure()
    plt.plot(rewards, marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("Rewards over Time")
    plt.grid(True)
    plt.show()

    # Close environment
    env.close()
