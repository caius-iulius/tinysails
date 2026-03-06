from environment import RegattaEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pygame
import numpy as np

pygame.init()
screen = pygame.display.set_mode((1000, 1000))
game_surf = pygame.Surface((1000, 1000))
game_rect = pygame.Rect((0,0,1000,1000))

clock = pygame.time.Clock()
x_zero = 500
y_zero = 500
scale = 10

def draw_boat(boat, surf):
    def boat_poly(posx, posy, angle):
        boatpoly = [(-10,-10), (-10,10), (20,0)]
        pos = pygame.math.Vector2(posx, posy)
        rotated_points = [
            pygame.math.Vector2(x, y).rotate(angle) + pos for x, y in boatpoly]
        return rotated_points

    pygame.draw.polygon(surf, (255,0,0), boat_poly(scale*boat.position[0]+x_zero,scale*boat.position[1]+y_zero, boat.heading_angle()*180/3.14159))
    sail_angle = boat.sail_angle(wind_vector) + boat.heading_angle()
    sail_mast = (scale*boat.position[0]+x_zero+13*boat.heading[0],scale*boat.position[1]+y_zero+13*boat.heading[1])
    pygame.draw.line(surf, (255,255,255), sail_mast, (sail_mast[0] - 25*np.cos(sail_angle), sail_mast[1] - 25*np.sin(sail_angle)), 2)

def draw_buoy(buoy, surf):
    color = (0, 255, 0) if buoy.passed else (255, 0, 0)
    pygame.draw.circle(surf, color, (int(scale*buoy.position[0]+x_zero), int(scale*buoy.position[1]+y_zero)), buoy.radius*scale)

def normalize_state(state):
    # Scale distance, angles, and speed to roughly [-1, 1]
    return [
        state[0] / 50.0,        # Max expected distance
        state[1] / 3.14159,     # Radians
        state[2] / 3.14159,     # Radians
        state[3] / 10.0         # Max expected speed
    ]

class ReinforcePolicy(nn.Module):
    def __init__(self):
        super(ReinforcePolicy, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

boat_params = {
    "mass": 3.0,
    "drag_coefficient": 1.0,
    "lift_coefficient": 45.0,
    "rotational_drag_coefficient": 3.0,
    "rudder_lift_coefficient": 2.0,
    "heading": [1, 0]
}
buoys = [[-30,0],[30,0],[0,30]]
wind_vector = [0, 10]

env = RegattaEnv(boat_params, buoys, wind_vector)
policy = ReinforcePolicy()
optimizer = optim.Adam(policy.parameters(), lr=0.002)

gamma = 0.99
num_episodes = 5000

for episode in range(num_episodes):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    tick_count = 0

    while not done and tick_count < 800:
        # Convert continuous state list to tensor
        state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32)

        # Network outputs probabilities for actions 0 and 1
        action_probs = policy(state_tensor)

        # Sample discrete action
        m = Categorical(action_probs)
        action = m.sample()

        # Step the environment
        # print(f"Episode {episode + 1:3d} | Tick {tick_count:4d} | Action taken: {action.item()-1}")
        next_state, reward, done = env.step(action.item()-1, 1/60)

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

    if (episode + 1) % 5 == 0:
        total_reward = sum(rewards)
        print(f"Episode {episode + 1:3d} | Total Reward: {total_reward:6.2f} | Ticks simulated: {tick_count}")

# game loop
running = True
last_time = pygame.time.get_ticks()
state = env.reset()
curr_score = 0
while running:
    current_time = pygame.time.get_ticks()
    time_step = (current_time - last_time) / 1000.0
    last_time = current_time

    print(f"Boat state: {state}, score: {curr_score:.2f}")
    state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32)
    action_probs = policy(state_tensor)
    action = torch.argmax(action_probs)
    env_action = action.item()-1
    print(f"action taken: {env_action}")

    state, reward, done = env.step(env_action, time_step)
    curr_score += reward


    game_surf.fill((0, 105, 205))
    for buoy in env.buoys:
        draw_buoy(buoy, game_surf)
    draw_boat(env.boat, game_surf)

    screen.blit(game_surf, game_rect)
    pygame.display.flip()

    if done:
        state = env.reset()
        curr_score = 0

    clock.tick(60)

pygame.quit()