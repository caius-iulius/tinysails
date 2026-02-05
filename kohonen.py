import numpy as np
import pygame

class EuclidKohonen:
    def __init__(self, map_shape, dimensions, learning_rate=1.0):
        self.learning_rate = learning_rate
        self.map_shape = map_shape
        self.dimensions = dimensions
        #self.weights = np.random.rand(*(*map_shape, dimensions))
        # TODO: cosa stracazzo fa sta cosa?
        grid_axes = [np.linspace(0, 1, num=s) for s in map_shape]
        grid = np.meshgrid(*grid_axes, indexing='ij')
        self.weights = np.stack(grid, axis=-1).reshape(*map_shape, dimensions)

    def get_bmu(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=-1)
        bmu_index = np.unravel_index(np.argmin(distances), self.map_shape)
        return bmu_index

    def update_weights(self, input_vector, bmu_index, iteration, max_iterations):
        learning_rate = self.learning_rate * (1 - iteration / max_iterations)
        learning_rate = self.learning_rate * np.exp(-iteration / (max_iterations / 3))
        bmu_pos = np.array(bmu_index)

        indices = np.indices(self.map_shape).transpose(tuple(
            (x+1)%(len(self.map_shape)+1) for x in range(len(self.map_shape)+1)
        ))
        relative_pos = indices - bmu_pos
        distance_to_bmu = np.linalg.norm(relative_pos, axis=-1, ord=1)
        neighborhood_function = np.exp(-distance_to_bmu)
        delta = learning_rate * neighborhood_function[..., np.newaxis] * (input_vector - self.weights)
        self.weights += delta

class DrawableKohonen(EuclidKohonen):
    def draw(self, surf):
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                weight = self.weights[i, j]
                x = int(weight[0] * surf.get_width())
                y = int(weight[1] * surf.get_height())
                pygame.draw.circle(surf, (255, 0, 0), (x, y), 5)