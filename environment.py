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
                print("Buoy passed!")

        return self.passed

    def reset(self):
        self.passed = False

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

class RegattaEnv:
    def __init__(self, boat_params, buoy_positions, wind_vector):
        self.boat = Boat(**boat_params)
        self.buoys = [Buoy(np.array(pos)) for pos in buoy_positions]
        self.current_buoy_index = 0
        self.wind_vector = wind_vector

    def reset(self):
        self.boat.reset()
        for buoy in self.buoys:
            buoy.reset()
        self.current_buoy_index = 0
        return self.boat.state()

    def step(self, action, dt):
        self.boat.update(self.wind_vector, action*np.pi/4, dt)

        reward = score_boat_tick(self.boat, self.buoys, self.current_buoy_index, dt)

        if self.current_buoy_index < len(self.buoys) and self.buoys[self.current_buoy_index].check(self.boat.position):
            self.current_buoy_index += 1

        done =  self.current_buoy_index >= len(self.buoys)

        return self.boat.state(), reward, done