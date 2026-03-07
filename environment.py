from boat_model import Boat
import numpy as np

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

        return self.passed

    def reset(self):
        self.passed = False

class RegattaEnv:
    def __init__(self, boat_params, buoy_positions, wind_vector):
        self.boat = Boat(**boat_params)
        self.buoys = [Buoy(np.array(pos)) for pos in buoy_positions]
        self.current_buoy_index = 0
        self.last_buoy_time = 0
        self.wind_vector = np.array(wind_vector)

    def reset(self):
        self.boat.reset()
        for buoy in self.buoys:
            buoy.reset()
        self.current_buoy_index = 0
        self.last_buoy_time = 0

        next_buoy_pos = self.buoys[0].position - self.boat.position
        next_buoy_distance = np.linalg.norm(next_buoy_pos)
        next_buoy_relative_angle = np.arctan2(next_buoy_pos[1], next_buoy_pos[0]) - np.arctan2(self.boat.heading[1], self.boat.heading[0])
        next_buoy_relative_angle = (next_buoy_relative_angle + np.pi) % (2 * np.pi) - np.pi
        relative_wind_angle = np.arctan2(self.wind_vector[1], self.wind_vector[0]) - np.arctan2(-self.boat.heading[1], -self.boat.heading[0])
        relative_wind_angle = (relative_wind_angle + np.pi) % (2 * np.pi) - np.pi

        return np.array([next_buoy_distance, next_buoy_relative_angle, relative_wind_angle, self.boat.speed, self.boat.rotational_velocity])

    def step(self, action, time, dt):
        self.boat.update(self.wind_vector, action*np.pi/4, dt)

        if self.current_buoy_index < len(self.buoys) and self.buoys[self.current_buoy_index].check(self.boat.position):
            self.last_buoy_time = time
            self.current_buoy_index += 1

        done =  self.current_buoy_index >= len(self.buoys)

        next_buoy_pos = self.buoys[self.current_buoy_index].position - self.boat.position if self.current_buoy_index < len(self.buoys) else np.array([0.0, 0.0])
        next_buoy_distance = np.linalg.norm(next_buoy_pos)
        next_buoy_relative_angle = np.arctan2(next_buoy_pos[1], next_buoy_pos[0]) - np.arctan2(self.boat.heading[1], self.boat.heading[0])
        next_buoy_relative_angle = (next_buoy_relative_angle + np.pi) % (2 * np.pi) - np.pi
        relative_wind_angle = np.arctan2(self.wind_vector[1], self.wind_vector[0]) - np.arctan2(-self.boat.heading[1], -self.boat.heading[0])
        relative_wind_angle = (relative_wind_angle + np.pi) % (2 * np.pi) - np.pi

        state = np.array([next_buoy_distance, next_buoy_relative_angle, relative_wind_angle, self.boat.speed, self.boat.rotational_velocity])

        return state, done