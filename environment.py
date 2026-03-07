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

def score_boat_tick(boat, buoys, current_buoy_idx, last_buoy_time, action, time, dt):
    # score proportional to relative velocity towards the next buoy, with a bonus for passing it. small score to absolute speed to encourage movement
    # score = 1 -2*abs(action)
    score = 0
    if current_buoy_idx >= len(buoys):
        return 0, False
    next_buoy = buoys[current_buoy_idx]
    to_buoy = next_buoy.position - boat.position
    distance = np.linalg.norm(to_buoy)

    relative_velocity = np.dot(boat.heading * boat.speed, to_buoy / distance)
    score += relative_velocity
    # score += boat.speed * 0.01 # encourage movement

    score *= dt # scale score by time step
    if distance < next_buoy.radius:
        score += 5000/(time - last_buoy_time) # bonus for passing buoy

    out_of_bounds = False
    # if abs(boat.position[0]) > 50 or abs(boat.position[1]) > 50:
    #     score -= 10
    #     out_of_bounds = True

    return score, out_of_bounds

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

        reward, oob = score_boat_tick(self.boat, self.buoys, self.current_buoy_index, self.last_buoy_time, action, time, dt)

        if self.current_buoy_index < len(self.buoys) and self.buoys[self.current_buoy_index].check(self.boat.position):
            self.last_buoy_time = time
            self.current_buoy_index += 1

        done =  self.current_buoy_index >= len(self.buoys) or oob

        next_buoy_pos = self.buoys[self.current_buoy_index].position - self.boat.position if self.current_buoy_index < len(self.buoys) else np.array([0.0, 0.0])
        next_buoy_distance = np.linalg.norm(next_buoy_pos)
        next_buoy_relative_angle = np.arctan2(next_buoy_pos[1], next_buoy_pos[0]) - np.arctan2(self.boat.heading[1], self.boat.heading[0])
        next_buoy_relative_angle = (next_buoy_relative_angle + np.pi) % (2 * np.pi) - np.pi
        relative_wind_angle = np.arctan2(self.wind_vector[1], self.wind_vector[0]) - np.arctan2(-self.boat.heading[1], -self.boat.heading[0])
        relative_wind_angle = (relative_wind_angle + np.pi) % (2 * np.pi) - np.pi

        state = np.array([next_buoy_distance, next_buoy_relative_angle, relative_wind_angle, self.boat.speed, self.boat.rotational_velocity])

        return state, reward, done