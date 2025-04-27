import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import random
from gymnasium.spaces import flatten_space
from gymnasium.spaces.utils import flatten

# <><><>><><><><><><><> SWAP THESE FOR CUSTOM EVALUATION<><><><><><><>#
from v1_1arm.V1env import JustDoItV1 
from DQNmodels import UNOarm, UNOarm_sign_based  # Import your network definition
# <><><>><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><><>#

model_dict_path = 'policy_state_dict_test.pth'

plt.ioff()  # Disable interactive mode to prevent extra windows

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env = JustDoItV1()

    flattened_obs_space = flatten_space(env.observation_space)
    state_dim = flattened_obs_space.shape[0]
    action_dim = env.action_space.n

    # Create your network using the computed state dimension.
    target = UNOarm(state_dim=state_dim, action_dim=action_dim, gridDim=30)
    target.load_state_dict(torch.load(model_dict_path, map_location=device))
    target.to(device)
    target.eval()

    obs, _ = env.reset()
    state = flatten(env.observation_space, obs)
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    accum_reward = 0
    rewards = []
    done = False
    epsilon = 0

    # Create a separate figure for reward animation
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, 1000)
    ax.set_ylim(min(-1, -10), max(1, 10))

    def animate(i):
        if i < len(rewards):
            line.set_data(range(i + 1), rewards[:i + 1])
        return line,

    for step in range(1000):
        with torch.no_grad():
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
                print("Random action chosen:", action)
            else:
                q_values = target(state_tensor)
                action = int(torch.argmax(q_values).item())
                print("Q-values:", q_values.cpu().numpy(), "Chosen greedy action:", action)

        obs, reward, done, info, _ = env.step(action)
        accum_reward += reward
        rewards.append(accum_reward)

        env.render()

        state = flatten(env.observation_space, obs)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        if done:
            print("Episode finished with reward:", accum_reward)
            break

    ani = FuncAnimation(fig, animate, frames=len(rewards), interval=50, blit=True)
    ani.save('reward_progress.gif', writer='pillow', fps=30)

    plt.close('all')  

    fig_static = plt.figure()
    plt.plot(rewards, marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title("Rewards over Time")
    plt.grid(True)
    plt.show()

    env.close()