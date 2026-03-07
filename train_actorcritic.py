from environment import RegattaEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import game_abstraction
import pygame
import numpy as np

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
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)

        self.actor_head = nn.Linear(16, 3)
        self.critic_head = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        # Get action probabilities
        action_logits = self.actor_head(x)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Get state value
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
buoys = [[-30,0],[0,-30],[0,30],[30,0],]
wind_vector = [0, 10]


def score_boat_tick(env, time, dt):
    score = 0.1*(env.current_buoy_index - len(env.buoys) + 1)
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

env = RegattaEnv(boat_params, buoys, wind_vector)
model = ActorCriticNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

gamma = 0.99
num_episodes = 100

for episode in range(num_episodes):
    state = env.reset()
    done = False
    tick_count = 0
    total_reward = 0

    while not done and tick_count < 1000:
        # 1. Evaluate current state
        state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32)
        action_probs, state_value = model(state_tensor)

        # 2. Sample action
        m = Categorical(action_probs)
        action = m.sample()

        # 3. Step environment
        next_state, done = env.step(action.item() - 1, tick_count * 0.1, 0.1)
        reward = score_boat_tick(env, tick_count * 0.1, 0.1)

        total_reward += reward

        # 4. Evaluate NEXT state (Bootstrapping)
        next_state_tensor = torch.tensor(normalize_state(next_state), dtype=torch.float32)
        _, next_state_value = model(next_state_tensor)

        # 5. Calculate TD Target and Advantage
        td_target = reward + gamma * next_state_value * (1 - int(done))
        advantage = td_target - state_value

        # --- THE FIX: Calculate Entropy ---
        # m is your Categorical(action_probs) distribution
        entropy = m.entropy()

        # This coefficient controls how aggressively we force exploration.
        # 0.05 is a good starting point for continuous/semi-continuous control.
        entropy_coef = 0.05

        # 6. Calculate Losses
        # We SUBTRACT the entropy bonus to lower the loss when the agent explores
        actor_loss = -m.log_prob(action) * advantage.detach() - (entropy_coef * entropy)

        # Critic loss remains the same
        critic_loss = nn.functional.mse_loss(state_value, td_target.detach())

        loss = actor_loss + critic_loss

        # 7. Update network immediately
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        tick_count += 1

    print(f"Episode {episode + 1:3d} | Total Reward: {total_reward:7.2f} | Ticks: {tick_count:3d} | Buoys: {env.current_buoy_index}")

input("Training complete. Press Enter to run inference...")

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

game_abstraction.run_game(env, inference)