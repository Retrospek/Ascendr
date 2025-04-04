import numpy as np
import matplotlib.pyplot as plt
import torch
from gymnasium.spaces.utils import flatten
from V1env import JustDoIt 
from DQNagent import DQN  # Ensure this imports your network definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load your trained model state (adjust the path as needed)
    target = DQN(state_dim=209, action_dim=4)  # Make sure to use proper state_dim and action_dim if required.
    target.load_state_dict(torch.load('policy_state_dict.pth', map_location=device))
    target.to(device)
    target.eval()

    env = JustDoIt()

    # Reset the environment and process the initial observation.
    obs, _ = env.reset()
    state = flatten(env.observation_space, obs)  # Flatten the observation
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    
    accum_reward = 0
    rewards = []
    done = False

    for _ in range(200):
        with torch.no_grad():
            # Pass the tensor state to the network
            q_values = target(state_tensor)
            action = int(torch.argmax(q_values).item())
            print("Q-values:", q_values.cpu().numpy(), "Chosen action:", action)
        # Take a step in the environment
        obs, reward, done, info, _ = env.step(action)
        accum_reward += reward
        rewards.append(accum_reward)

        env.render()  # Render the environment

        # Prepare the next state
        state = flatten(env.observation_space, obs)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

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
