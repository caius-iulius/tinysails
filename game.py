import pygame
import boat_model
import numpy as np

# Initialize Pygame and set up the display
pygame.init()
screen = pygame.display.set_mode((1000, 1000))
clock = pygame.time.Clock()
x_zero = 500
y_zero = 500
scale = 30

wind_vector = np.array([0, 10])
boat = boat_model.Boat(mass=3.0, drag_coefficient=1.0, lift_coefficient=45.0, rotational_drag_coefficient=3.0, rudder_lift_coefficient=2.0)
deltaheading = 0

boatpoly = [(-10,-10), (-10,10), (20,0)]

def boat_poly(posx, posy, angle):
    pos = pygame.math.Vector2(posx, posy)
    rotated_points = [
        pygame.math.Vector2(x, y).rotate(angle) + pos for x, y in boatpoly]
    return rotated_points

def show_boat():
    pygame.draw.polygon(screen, (255,0,0), boat_poly(scale*boat.position[0]+x_zero,scale*boat.position[1]+y_zero, boat.heading_angle()*180/3.14159))
    sail_angle = boat.sail_angle(wind_vector) + boat.heading_angle()
    sail_mast = (scale*boat.position[0]+x_zero+13*boat.heading[0],scale*boat.position[1]+y_zero+13*boat.heading[1])
    pygame.draw.line(screen, (255,255,255), sail_mast, (sail_mast[0] - 25*np.cos(sail_angle), sail_mast[1] - 25*np.sin(sail_angle)), 2)

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

    def draw(self, screen):
        color = (0, 255, 0) if self.passed else (255, 0, 0)
        pygame.draw.circle(screen, color, (int(scale*self.position[0]+x_zero), int(scale*self.position[1]+y_zero)), self.radius*scale)

buoys = [Buoy(np.array([0,-30])), Buoy(np.array([30,0])), Buoy(np.array([0,30]))]
current_buoy_index = 0

# Run the game loop
running = True
last_time = pygame.time.get_ticks()
while running:
    # Handle user input
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

    # Use actual time step for consistent movement speed
    current_time = pygame.time.get_ticks()
    time_step = (current_time - last_time) / 1000.0
    last_time = current_time
    boat.update(wind_vector, deltaheading*np.pi/4, time_step) #TODO: timestep unchanged, fix params
    if current_buoy_index < len(buoys) and buoys[current_buoy_index].check(boat.position):
        current_buoy_index += 1
        if current_buoy_index >= len(buoys):
            print("All buoys passed! Race finished.")

    print(f"Boat position: {boat.position}, speed: {boat.speed}, heading: {boat.heading}, sail angle: {boat.sail_angle(wind_vector)}, current buoy: {current_buoy_index}")

    # Draw the ball and update the display
    screen.fill((0, 105, 205))
    for buoy in buoys:
        buoy.draw(screen)
    show_boat()
    pygame.display.flip()
    clock.tick(60)
# Quit Pygame when the game loop is finished
pygame.quit()