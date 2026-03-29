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

class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()
        # Shared feature extractor
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)

        self.actor_head = nn.Linear(32, 3)
        self.critic_head = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        action_logits = self.actor_head(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        state_value = self.critic_head(x)

        return action_probs, state_value

# reward function
def score_boat_tick(env, time, dt):
    score = 0.1*(env.current_buoy_index - len(env.buoys) + 1)
    #score = -0.5*(1 - env.current_buoy_index/len(env.buoys))
    if env.current_buoy_index >= len(env.buoys):
        return 0
    next_buoy = env.buoys[env.current_buoy_index]
    to_buoy = next_buoy.position - env.boat.position
    distance = np.linalg.norm(to_buoy)

    relative_velocity = np.dot(env.boat.heading * env.boat.speed, to_buoy / distance)
    score += 0.5*relative_velocity

    score *= dt # scale score by time step
    if distance < next_buoy.radius:
        score += 50

    return score

# hyperparameters
gamma = 0.99
num_episodes = 2000
num_seeds = 5
lr_start = 0.0005
lr_end = 5e-5

entropy_coef_start = 0.10
entropy_coef_end   = 0.005

# physics params
boat_params = {
    "mass": 3.0,
    "drag_coefficient": 1.0,
    "lift_coefficient": 45.0,
    "rotational_drag_coefficient": 3.0,
    "rudder_lift_coefficient": 2.0,
    "heading": [1, 0]
}
buoys = [[-30,0],[0,-30],[0,25],[30,0],[0,0],[0,30],[0,-25]]
wind_vector = [0, 10]
env = RegattaEnv(boat_params, buoys, wind_vector)

logs = []

for seed in range(num_seeds):
    print(f"--- Training Seed {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ActorCriticNetwork()
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_episodes, eta_min=lr_end
    )

    # training
    best_ticks = 1500
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        tick_count = 0
        total_reward = 0

        progress = episode / max(num_episodes - 1, 1)
        entropy_coef = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * progress

        while not done and tick_count < 1500:
            state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32)
            action_probs, state_value = model(state_tensor)

            m = Categorical(action_probs)
            action = m.sample()

            next_state, done = env.step(action.item() - 1, tick_count * 0.1, 0.1)
            reward = score_boat_tick(env, tick_count * 0.1, 0.1)
            total_reward += reward

            # bootstrap
            next_state_tensor = torch.tensor(normalize_state(next_state), dtype=torch.float32)
            _, next_state_value = model(next_state_tensor)

            td_target = reward + gamma * next_state_value * (1 - int(done))
            advantage = td_target - state_value

            # entropy bonus
            entropy = m.entropy()

            # losses
            actor_loss = -m.log_prob(action) * advantage.detach() - entropy_coef * entropy
            critic_loss = nn.functional.mse_loss(state_value, td_target.detach())
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            tick_count += 1

        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        progress    = episode / max(num_episodes - 1, 1)
        entropy_coef_display = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * progress
        print(f"Seed {seed} | Episode {episode + 1:3d} | Total Reward: {total_reward:7.2f} | Ticks: {tick_count:3d} | Buoys: {env.current_buoy_index} | LR: {current_lr:.6f} | Ent: {entropy_coef_display:.4f}")
        logs.append({
            "seed": seed,
            "episode": episode + 1,
            "total_reward": total_reward,
            "ticks": tick_count,
            "buoys_passed": env.current_buoy_index,
            "learning_rate": current_lr,
            "entropy_coef": entropy_coef_display
        })

        if tick_count < best_ticks:
            best_ticks = tick_count
            print(f"New best time for seed {seed}: {best_ticks} ticks")
            torch.save(model.state_dict(), f"./models/actorcritic_model_seed_{seed}.pth")
            torch.save(model.state_dict(), "./models/actorcritic_model.pth")

with open("./logs/actorcritic.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=logs[0].keys())
    writer.writeheader()
    writer.writerows(logs)

input("Training complete. Press Enter to run inference...")

model2 = ActorCriticNetwork()
model2.load_state_dict(torch.load("./models/actorcritic_model.pth"))

def inference(state):
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32)

    action_probs, _ = model2(state_tensor)

    action = torch.argmax(action_probs)
    env_action = action.item() - 1
    return env_action, running

env.reset()
game_abstraction.run_game(env, inference)