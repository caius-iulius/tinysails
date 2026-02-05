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

boat_params = {
    "mass": 3.0,
    "drag_coefficient": 1.0,
    "lift_coefficient": 45.0,
    "rotational_drag_coefficient": 3.0,
    "rudder_lift_coefficient": 2.0
}
wind_vector = np.array([0, 10])
boat = GameBoat(**boat_params)

buoys = [Buoy(np.array([0,-30])), Buoy(np.array([30,0])), Buoy(np.array([0,30]))]
current_buoy_index = 0

kohonen = DrawableKohonen(map_shape=(3, 10), dimensions=2, learning_rate=0.1)
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

    datapoint = np.array([(boat.speed + x_bias) / x_norm, (boat.heading_angle() + y_bias) / y_norm])
    kohonen.update_weights(datapoint, kohonen.get_bmu(datapoint), len(dataset), 10000)
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