import numpy as np
import pygame
from kohonen import DrawableKohonen

net = DrawableKohonen(map_shape=(10, 10), dimensions=2, learning_rate=1)
data = np.random.rand(1000, 2) * 0.5 + 0.25 # random points in the center of the screen
data2= np.random.rand(200, 2) * 0.2 + 0.75
data3= np.random.rand(200, 2) * 0.2 + 0.05
data = np.concatenate((data, data2, data3), axis=0)

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
running = True
iteration = 0
max_iterations = 1000

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    # Draw dataset points
    for point in data:
        x = int(point[0] * 800)
        y = int(point[1] * 600)
        pygame.draw.circle(screen, (0, 255, 0), (x, y), 3)

    # Draw network weights
    net.draw(screen)

    pygame.display.flip()

    # Update network
    if iteration < max_iterations:
        print("Iteration:", iteration)

        input_vector = data[np.random.randint(0, len(data))]
        bmu_index = net.get_bmu(input_vector)
        net.update_weights(input_vector, bmu_index, iteration, max_iterations)

        # for input_vector in data:
        #     bmu_index = net.get_bmu(input_vector)
        #     net.update_weights(input_vector, bmu_index, iteration, max_iterations)

        iteration += 1
    elif iteration == max_iterations:
        print(f"Training complete. Performance: {1000*max_iterations/pygame.time.get_ticks()} it/s")
        iteration += 1

    #clock.tick(60)