import pygame
import numpy as np
import environment

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

boat_params = {
    "mass": 3.0,
    "drag_coefficient": 1.0,
    "lift_coefficient": 45.0,
    "rotational_drag_coefficient": 3.0,
    "rudder_lift_coefficient": 2.0
}
buoys = [[0,-30],[30,0],[0,30]]
wind_vector = [0, 10]

env = environment.RegattaEnv(boat_params, buoys, wind_vector)
curr_score = 0

# game loop
running = True
last_time = pygame.time.get_ticks()
deltaheading = 0
while running:
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

    current_time = pygame.time.get_ticks()
    time_step = (current_time - last_time) / 1000.0
    last_time = current_time

    state, reward, done = env.step(deltaheading, time_step)
    curr_score += reward

    # next_buoy_distance = np.linalg.norm(boat.position - buoys[current_buoy_index].position) if current_buoy_index < len(buoys) else 0
    # next_buoy_relative_angle = np.arctan2(buoys[current_buoy_index].position[1] - boat.position[1], buoys[current_buoy_index].position[0] - boat.position[0]) - boat.heading_angle() if current_buoy_index < len(buoys) else 0
    # print(f"Next buoy distance: {next_buoy_distance:2f}, relative angle: {next_buoy_relative_angle:2f}, score: {curr_score:.2f}")

    print(f"Boat state: {state}, score: {curr_score:.2f}")

    game_surf.fill((0, 105, 205))
    for buoy in env.buoys:
        draw_buoy(buoy, game_surf)
    draw_boat(env.boat, game_surf)

    screen.blit(game_surf, game_rect)
    pygame.display.flip()

    if done:
        env.reset()
        curr_score = 0

    clock.tick(60)

pygame.quit()