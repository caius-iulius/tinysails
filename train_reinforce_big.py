from environment import RegattaEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import game_abstraction
import pygame

def normalize_state(state):
    # Scale distance, angles, and speed to roughly [-1, 1]
    return [
        state[0] / 50.0,        # Max expected distance
        state[1] / 3.14159,     # Radians
        state[2] / 3.14159,     # Radians
        state[3] / 10.0,         # Max expected speed
        state[4] # rotational velocity
    ]

class ReinforcePolicy(nn.Module):
    def __init__(self):
        super(ReinforcePolicy, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return torch.softmax(x, dim=-1)

boat_params = {
    "mass": 3.0,
    "drag_coefficient": 1.0,
    "lift_coefficient": 45.0,
    "rotational_drag_coefficient": 3.0,
    "rudder_lift_coefficient": 2.0,
    "heading": [1, 0]
}
buoys = [[-30,0],[0,-30],[0,30],[30,0],]
wind_vector = [0, 10]

env = RegattaEnv(boat_params, buoys, wind_vector)
policy = ReinforcePolicy()
optimizer = optim.Adam(policy.parameters(), lr=0.0003)

gamma = 0.99
num_episodes = 8000

for episode in range(num_episodes):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    tick_count = 0

    while not done and tick_count < 1000:
        # Convert continuous state list to tensor
        state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32)

        # Network outputs probabilities for actions 0 and 1
        action_probs = policy(state_tensor)

        # Sample discrete action
        m = Categorical(action_probs)
        action = m.sample()

        # Step the environment
        # print(f"Episode {episode + 1:3d} | Tick {tick_count:4d} | Action taken: {action.item()-1}")
        next_state, reward, done = env.step(action.item()-1, tick_count*0.1, 0.1)

        log_probs.append(m.log_prob(action))
        rewards.append(reward)

        state = next_state
        tick_count += 1

    # Calculate returns
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    # Update policy
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()

    # if (episode + 1) % 5 == 0:
    total_reward = sum(rewards)
    print(f"Episode {episode + 1:3d} | Total Reward: {total_reward:7.2f} | Ticks simulated: {tick_count:3d} | Buoys passed: {env.current_buoy_index}")

def inference(state):
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32)
    action_probs = policy(state_tensor)
    action = torch.argmax(action_probs)
    env_action = action.item()-1
    return env_action, running

game_abstraction.run_game(env, inference)