import pygame
import boat_model
import numpy as np
from kohonen import DrawableKohonen

pygame.init()
screen = pygame.display.set_mode((2000, 1000))
game_surf = pygame.Surface((1000, 1000))
game_rect = pygame.Rect((0,0,1000,1000))
kohonen_surf = pygame.Surface((1000, 1000))
kohonen_rect = pygame.Rect((1000,0,1000,1000))

clock = pygame.time.Clock()
x_zero = 500
y_zero = 500
scale = 10

class GameBoat(boat_model.Boat):
    def draw(self, surf):
        def boat_poly(posx, posy, angle):
            boatpoly = [(-10,-10), (-10,10), (20,0)]
            pos = pygame.math.Vector2(posx, posy)
            rotated_points = [
                pygame.math.Vector2(x, y).rotate(angle) + pos for x, y in boatpoly]
            return rotated_points

        pygame.draw.polygon(surf, (255,0,0), boat_poly(scale*self.position[0]+x_zero,scale*self.position[1]+y_zero, self.heading_angle()*180/3.14159))
        sail_angle = self.sail_angle(wind_vector) + self.heading_angle()
        sail_mast = (scale*self.position[0]+x_zero+13*self.heading[0],scale*self.position[1]+y_zero+13*self.heading[1])
        pygame.draw.line(surf, (255,255,255), sail_mast, (sail_mast[0] - 25*np.cos(sail_angle), sail_mast[1] - 25*np.sin(sail_angle)), 2)

class Buoy:
    def __init__(self, position):
        self.position = position
        self.radius = 1
        self.passed = False

    def check(self, boat_position):
        if not self.passed:
            distance = np.linalg.norm(boat_position - self.position)
            if distance <= self.radius:
                self.passed = True
                print("Buoy passed!")

        return self.passed

    def draw(self, surf):
        color = (0, 255, 0) if self.passed else (255, 0, 0)
        pygame.draw.circle(surf, color, (int(scale*self.position[0]+x_zero), int(scale*self.position[1]+y_zero)), self.radius*scale)

def score_boat_tick(boat, buoys, current_buoy_idx, dt):
    # score proportional to relative velocity towards the next buoy, with a bonus for passing it. small score to absolute speed to encourage movement
    score = -1 # small negative score to encourage faster completion
    if current_buoy_idx >= len(buoys):
        return 0
    next_buoy = buoys[current_buoy_idx]
    to_buoy = next_buoy.position - boat.position
    distance = np.linalg.norm(to_buoy)

    relative_velocity = np.dot(boat.heading * boat.speed, to_buoy / distance)
    score += relative_velocity / (distance + 1) # avoid division by zero
    score += boat.speed * 0.05 # encourage movement

    score *= dt # scale score by time step
    if distance < next_buoy.radius:
        score += 10 # bonus for passing buoy

    return score

boat_params = {
    "mass": 3.0,
    "drag_coefficient": 1.0,
    "lift_coefficient": 45.0,
    "rotational_drag_coefficient": 3.0,
    "rudder_lift_coefficient": 2.0
}
wind_vector = np.array([0, 10])
boat = GameBoat(**boat_params)
curr_score = 0

buoys = [Buoy(np.array([0,-30])), Buoy(np.array([30,0])), Buoy(np.array([0,30]))]
current_buoy_index = 0

kohonen = DrawableKohonen(map_shape=(3, 10), dimensions=2, learning_rate=0.04)
dataset = []
x_norm = 2*np.linalg.norm(wind_vector)
x_bias = 0
y_norm = 2*np.pi
y_bias = np.pi

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

    boat.update(wind_vector, deltaheading*np.pi/4, time_step)
    curr_score += score_boat_tick(boat, buoys, current_buoy_index, time_step)

    datapoint = np.array([(boat.speed + x_bias) / x_norm, (boat.heading_angle() + y_bias) / y_norm])
    kohonen.update_weights(datapoint, kohonen.get_bmu(datapoint), min(len(dataset),10000), 10000)
    dataset.append(datapoint)

    kohonen_surf.fill((0, 0, 0))
    # draw dataset points
    for point in dataset:
        x = int(point[0] * 1000)
        y = int(point[1] * 1000)
        pygame.draw.circle(kohonen_surf, (0, 255, 0), (x, y), 2)
    kohonen.draw(kohonen_surf)

    if current_buoy_index < len(buoys) and buoys[current_buoy_index].check(boat.position):
        current_buoy_index += 1
        if current_buoy_index >= len(buoys):
            print("All buoys passed! Race finished. Time:", current_time/1000.0, "seconds")
            break

    next_buoy_distance = np.linalg.norm(boat.position - buoys[current_buoy_index].position) if current_buoy_index < len(buoys) else 0
    next_buoy_relative_angle = np.arctan2(buoys[current_buoy_index].position[1] - boat.position[1], buoys[current_buoy_index].position[0] - boat.position[0]) - boat.heading_angle() if current_buoy_index < len(buoys) else 0

    print(f"Next buoy distance: {next_buoy_distance:2f}, relative angle: {next_buoy_relative_angle:2f}, score: {curr_score:.2f}")
    print(f"Boat position: {boat.position}, speed: {boat.speed}, heading: {boat.heading}, sail angle: {boat.sail_angle(wind_vector)}, current buoy: {current_buoy_index}")

    game_surf.fill((0, 105, 205))
    for buoy in buoys:
        buoy.draw(game_surf)
    boat.draw(game_surf)

    screen.blit(game_surf, game_rect)
    screen.blit(kohonen_surf, kohonen_rect)
    pygame.display.flip()

    clock.tick(60)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()