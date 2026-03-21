from environment import RegattaEnv
import torch
import torch.nn as nn
import game_abstraction
import pygame

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

wind_vector = [0, 10]

env = RegattaEnv(boat_params, [[-30,0],[0,-30],[0,25],[30,0],[0,0],[0,30],[0,-25]], wind_vector)
# env.random_buoys(7)

model = ActorCriticNetwork()
model.load_state_dict(torch.load("./models/actorcritic_model.pth"))

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