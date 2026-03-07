import pygame
import environment
import game_abstraction

boat_params = {
    "mass": 3.0,
    "drag_coefficient": 1.0,
    "lift_coefficient": 45.0,
    "rotational_drag_coefficient": 3.0,
    "rudder_lift_coefficient": 2.0
}
buoys = [[-30,0],[0,-30],[0,25],[30,0],[0,0],[0,30],[0,-25]]
wind_vector = [0, 10]

env = environment.RegattaEnv(boat_params, buoys, wind_vector)

deltaheading = 0
def get_command(state):
    global deltaheading
    running = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                deltaheading = 1
            elif event.key == pygame.K_RIGHT:
                deltaheading = -1
        elif event.type == pygame.KEYUP:
            deltaheading = 0

    return deltaheading, running

game_abstraction.run_game(env, get_command)