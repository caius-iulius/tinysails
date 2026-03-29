from environment import RegattaEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import game_abstraction
import pygame
import numpy as np
import csv

# model
def normalize_state(state):
    return [
        state[0] / 50.0, # next buoy dist
        state[1], # next buoy sin
        state[2], # next buoy cos
        state[3], # wind sin
        state[4], # wind cos
        state[5] / 10.0, # boat speed
        state[6] # rotational velocity
    ]

class ReinforcePolicy(nn.Module):
    def __init__(self):
        super(ReinforcePolicy, self).__init__()
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return torch.softmax(x, dim=-1)

# reward function
def score_boat_tick(env, time, dt):
    score = 0
    if env.current_buoy_index >= len(buoys):
        return 0
    next_buoy = env.buoys[env.current_buoy_index]
    to_buoy = next_buoy.position - env.boat.position
    distance = np.linalg.norm(to_buoy)

    relative_velocity = np.dot(env.boat.heading * env.boat.speed, to_buoy / distance)
    score += relative_velocity

    score *= dt
    if distance < next_buoy.radius:
        score += 5000/(time - env.last_buoy_time) # bonus for passing buoy

    return score

# hyperparameters
lr=0.0003
gamma = 0.99
num_episodes = 4000
num_seeds = 5

# physics params
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

logs = []

for seed in range(num_seeds):
    print(f"--- Training Seed {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = ReinforcePolicy()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        tick_count = 0

        while not done and tick_count < 1000:
            state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32)
            action_probs = policy(state_tensor)

            m = Categorical(action_probs)
            action = m.sample()

            next_state, done = env.step(action.item()-1, tick_count*0.1, 0.1)
            reward = score_boat_tick(env, tick_count*0.1, 0.1)

            log_probs.append(m.log_prob(action))
            rewards.append(reward)

            state = next_state
            tick_count += 1

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        print(f"Seed {seed} | Episode {episode + 1:3d} | Total Reward: {total_reward:7.2f} | Ticks: {tick_count:3d} | Buoys: {env.current_buoy_index}")
        logs.append({
            "seed": seed,
            "episode": episode + 1,
            "total_reward": total_reward,
            "ticks": tick_count,
            "buoys_passed": env.current_buoy_index
        })

with open('logs/reinforce_big.csv', 'w', newline='') as csvfile:
    fieldnames = ['seed', 'episode', 'total_reward', 'ticks', 'buoys_passed']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for log in logs:
        writer.writerow(log)

input("Training complete. Press Enter to run inference...")

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

env.reset()
game_abstraction.run_game(env, inference)