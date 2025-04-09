import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from gymnasium.spaces import flatten_space
from gymnasium.spaces.utils import flatten
from V1env import JustDoIt 
from DQNagent import DQN  # Import your network definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env = JustDoIt()
    
    flattened_obs_space = flatten_space(env.observation_space)
    state_dim = flattened_obs_space.shape[0]
    action_dim = env.action_space.n  

    # Create your network using the computed state dimension.
    # Note: gridDim is set to 50 as per your environment.
    target = DQN(state_dim=state_dim, action_dim=action_dim, gridDim=50)
    
    target.load_state_dict(torch.load('policy_state_dict.pth', map_location=device))
    target.to(device)
    target.eval()

    obs, _ = env.reset()
    state = flatten(env.observation_space, obs)  # Flatten the observation
    # Add a batch dimension so that our network receives input of shape [1, state_dim]
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    accum_reward = 0
    rewards = []
    done = False

    epsilon = 0.1  # Adjust as needed

    for step in range(200):
        with torch.no_grad():
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
                print("Random action chosen:", action)
            else:
                q_values = target(state_tensor)
                action = int(torch.argmax(q_values).item())
                print("Q-values:", q_values.cpu().numpy(), "Chosen greedy action:", action)

        # Step in the environment.
        # Note: your environment's step returns (obs, reward, done, info, _)
        obs, reward, done, info, _ = env.step(action)
        accum_reward += reward
        rewards.append(accum_reward)

        # Render the environment state.
        env.render()

        state = flatten(env.observation_space, obs)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        if done:
            print("Episode finished with reward:", accum_reward)
            break

    # Plot the cumulative reward over time.
    plt.ioff()  
    plt.figure()
    plt.plot(rewards, marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title("Rewards over Time")
    plt.grid(True)
    plt.show()

    env.close()
