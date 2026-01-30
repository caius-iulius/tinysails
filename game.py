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
wind_vector = np.array([0, 1.0])

heading = np.pi
deltaheading = 0

boatpoly = [(-10,-10), (-10,10), (20,0)]

def boat_poly(posx, posy, angle):
    pos = pygame.math.Vector2(posx, posy)
    rotated_points = [
        pygame.math.Vector2(x, y).rotate(angle) + pos for x, y in boatpoly]
    return rotated_points

boat = boat_model.Boat(mass=1.0, drag_coefficient=0.5, lift_coefficient=2.0, rudder_lift_coefficient=0.5)

def show_boat():
    pygame.draw.polygon(screen, (255,0,0), boat_poly(scale*boat.position[0]+x_zero,scale*boat.position[1]+y_zero, boat.heading_angle()*180/3.14159))
    sail_angle = boat.sail_angle(np.array([0, 1.0])) + boat.heading_angle()
    sail_mast = (scale*boat.position[0]+x_zero+0.4*scale*boat.heading[0],scale*boat.position[1]+y_zero+0.4*scale*boat.heading[1])
    pygame.draw.line(screen, (255,255,255), sail_mast, (sail_mast[0] - 0.8*scale*np.cos(sail_angle), sail_mast[1] - 0.8*scale*np.sin(sail_angle)), 2)

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
    boat.update(wind_vector, deltaheading*np.pi/4, time_step*6) #TODO: timestep unchanged, fix params

    #print(f"Boat position: {boat.position}, speed: {boat.speed}, heading: {boat.heading}, sail angle: {boat.sail_angle(wind_vector)}")

    # Draw the ball and update the display
    screen.fill((0, 105, 205))
    show_boat()
    pygame.display.flip()
    clock.tick(60)
# Quit Pygame when the game loop is finished
pygame.quit()