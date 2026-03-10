from boat_model import Boat
import numpy as np
from random import randrange

def gen_random_buoys(n_buoys):
    buoys = []
    while len(buoys) < n_buoys:
        new_buoy = (randrange(-30, 30), randrange(-30, 30))
        if len(buoys) == 0 or (buoys[-1][0]-new_buoy[0])**2 + (buoys[-1][1]-new_buoy[1])**2 > 1000:
            buoys.append(new_buoy)
    return buoys

class Buoy:
    def __init__(self, position, label=None):
        self.position = position
        self.radius = 1
        self.passed = False
        self.label = label

    def check(self, boat_position):
        if not self.passed:
            distance = np.linalg.norm(boat_position - self.position)
            if distance <= self.radius:
                self.passed = True

        return self.passed

    def reset(self):
        self.passed = False

class RegattaEnv:
    def __init__(self, boat_params, buoy_positions, wind_vector):
        self.boat = Boat(**boat_params)
        self.buoys = [Buoy(np.array(pos), str(idx+1)) for idx,pos in enumerate(buoy_positions)]
        self.current_buoy_index = 0
        self.last_buoy_time = 0
        self.wind_vector = np.array(wind_vector)

    def set_buoys(self, buoy_positions):
        self.buoys = [Buoy(np.array(pos), str(idx+1)) for idx,pos in enumerate(buoy_positions)]
        self.current_buoy_index = 0
        self.last_buoy_time = 0

    def random_buoys(self, n_buoys):
        self.set_buoys(gen_random_buoys(n_buoys))

    def state(self):
        next_buoy_pos = self.buoys[self.current_buoy_index].position - self.boat.position if self.current_buoy_index < len(self.buoys) else np.array([0.0, 0.0])
        next_buoy_dist = np.linalg.norm(next_buoy_pos)
        next_buoy_normalized = next_buoy_pos / next_buoy_dist if next_buoy_dist else np.array([0.0, 0.0])

        next_buoy_sin = np.dot(next_buoy_normalized, np.array([-self.boat.heading[1], self.boat.heading[0]]))
        next_buoy_cos = np.dot(next_buoy_normalized, self.boat.heading)
        wind_sin = np.dot(self.wind_vector, np.array([-self.boat.heading[1], self.boat.heading[0]])) / np.linalg.norm(self.wind_vector)
        wind_cos = np.dot(self.wind_vector, self.boat.heading) / np.linalg.norm(self.wind_vector)

        return np.array([next_buoy_dist, next_buoy_sin, next_buoy_cos, wind_sin, wind_cos, self.boat.speed, self.boat.rotational_velocity])

    def reset(self):
        self.boat.reset()
        for buoy in self.buoys:
            buoy.reset()
        self.current_buoy_index = 0
        self.last_buoy_time = 0

        return self.state()

    def step(self, action, time, dt):
        self.boat.update(self.wind_vector, action*np.pi/4, dt)

        if self.current_buoy_index < len(self.buoys) and self.buoys[self.current_buoy_index].check(self.boat.position):
            self.last_buoy_time = time
            self.current_buoy_index += 1

        done =  self.current_buoy_index >= len(self.buoys)

        return self.state(), done