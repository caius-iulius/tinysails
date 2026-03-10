from environment import RegattaEnv, gen_random_buoys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import game_abstraction
import pygame
import numpy as np
import time

def normalize_state(state):
    # Scale distance, angles, and speed to roughly [-1, 1]
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

boat_params = {
    "mass": 3.0,
    "drag_coefficient": 1.0,
    "lift_coefficient": 45.0,
    "rotational_drag_coefficient": 3.0,
    "rudder_lift_coefficient": 2.0,
    "heading": [1, 0]
}
wind_vector = [0, 10]
env = RegattaEnv(boat_params, [], wind_vector)

def score_boat_tick(env, time, dt):
    score = 0.2*(env.current_buoy_index - len(env.buoys) + 1)
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

model = ActorCriticNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.003)

gamma = 0.99
num_episodes = 1000
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_episodes, eta_min=1e-5
)

entropy_coef_start = 0.10
entropy_coef_end   = 0.000

best_ticks = 1500
batch_size = 32

training_time = time.time()
for episode in range(num_episodes):
    env.random_buoys(7)
    state = env.reset()
    done = False
    tick_count = 0
    total_reward = 0

    # Calculate entropy coefficient once per episode
    progress = episode / max(num_episodes - 1, 1)
    entropy_coef = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * progress

    while not done and tick_count < 1500:
        # 1. Initialize lists to store our trajectory batch
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropies = []

        # 2. Collect N steps of data
        for _ in range(batch_size):
            if done or tick_count >= 1500:
                break

            state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32)
            action_probs, state_value = model(state_tensor)

            m = Categorical(action_probs)
            action = m.sample()

            next_state, done = env.step(action.item() - 1, tick_count * 0.1, 0.1)
            reward = score_boat_tick(env, tick_count * 0.1, 0.1)

            total_reward += reward

            # Save step data
            log_probs.append(m.log_prob(action))
            values.append(state_value)
            rewards.append(reward)
            # Mask is 0 if the race finished, 1 otherwise
            masks.append(1.0 - float(done))
            entropies.append(m.entropy())

            state = next_state
            tick_count += 1

        # 3. Evaluate the NEXT state to bootstrap the final value
        next_state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32)
        _, next_state_value = model(next_state_tensor)
        R = next_state_value.item() * (1.0 - float(done))

        # 4. Calculate N-Step Returns backwards
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)

        # 5. Convert lists to tensors and FORCE 1D shapes to prevent broadcasting
        returns_tensor = torch.tensor(returns, dtype=torch.float32).view(-1)
        values_tensor = torch.cat(values).view(-1)
        log_probs_tensor = torch.stack(log_probs).view(-1)
        entropies_tensor = torch.stack(entropies).view(-1)

        # 6. Calculate Advantages
        advantages = returns_tensor - values_tensor
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 7. Calculate Batched Losses
        actor_loss = -(log_probs_tensor * advantages.detach()).mean()
        critic_loss = nn.functional.mse_loss(values_tensor, returns_tensor.detach())
        entropy_loss = entropies_tensor.mean()

        # Added the 0.5 value loss coefficient to balance the gradients
        loss = actor_loss + 0.5*critic_loss - (entropy_coef * entropy_loss)

        # 8. Update network
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # gradient clipping
        optimizer.step()

    # Episode cleanup & Logging
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Episode {episode + 1:3d} | Total Reward: {total_reward:7.2f} | Ticks: {tick_count:3d} | Buoys: {env.current_buoy_index} | LR: {current_lr:.6f} | Ent: {entropy_coef_start + (entropy_coef_end - entropy_coef_start) * progress:.4f}")

training_time = time.time() - training_time
torch.save(model.state_dict(), "./models/actorcritic_model.pth")
input(f"Training completed in {training_time} seconds. Press Enter to run inference...")

def inference(state):
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32)

    action_probs, _ = model(state_tensor)

    action = torch.argmax(action_probs)
    env_action = action.item() - 1
    return env_action, running

env.random_buoys(7)
game_abstraction.run_game(env, inference)