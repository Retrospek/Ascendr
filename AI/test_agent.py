import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import random
from gymnasium.spaces import flatten_space
from gymnasium.spaces.utils import flatten
from v1_1arm.V1env import JustDoItV1
from DQNmodels import UNOarm_sign_based

plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dict_path = 'AI/v1_1arm/v2_models/target_state_dict_test_v2.pth'

if __name__ == "__main__":
    env = JustDoItV1()
    flattened_obs_space = flatten_space(env.observation_space)
    state_dim = flattened_obs_space.shape[0]
    action_dim = env.action_space.n

    target = UNOarm_sign_based(state_dim=state_dim, action_dim=action_dim, gridDim=30).to(device)
    target.load_state_dict(torch.load(model_dict_path, map_location=device))
    target.eval()

    obs, _ = env.reset()
    state = flatten(env.observation_space, obs)
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    accum_reward = 0
    rewards = []
    epsilon = 0

    for step in range(555):
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                q_values = target(state_tensor)
                action = int(torch.argmax(q_values).item())

        obs, reward, terminated, truncated, info = env.step(action)
        accum_reward += reward
        rewards.append(accum_reward)

        env.render()
        state = flatten(env.observation_space, obs)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        if terminated:
            print(f"Episode finished with reward: {accum_reward}")
            break

    fig, ax = plt.subplots()
    ax.set_xlim(0, len(rewards) - 1)
    ax.set_ylim(min(rewards), max(rewards))
    line, = ax.plot([], [], lw=2)

    def animate(i):
        line.set_data(range(i + 1), rewards[:i + 1])
        return line,

    ani = FuncAnimation(fig, animate, frames=len(rewards), interval=200, blit=True)
    ani.save('reward_progress.gif', writer='pillow', fps=30)

    plt.close(fig)
    fig_static, ax_static = plt.subplots()
    ax_static.plot(rewards, marker='o')
    ax_static.set_xlabel("Steps")
    ax_static.set_ylabel("Cumulative Reward")
    ax_static.set_title("Rewards over Time")
    ax_static.grid(True)
    plt.show()
    env.close()
