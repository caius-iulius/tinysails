import numpy as np
from numpy import dot

def relative_sail_efficiency(wind_vector, boat_versor):
    efficiency = 0.5 * (np.linalg.norm(wind_vector) + dot(wind_vector, boat_versor))
    return efficiency

class Boat:
    def __init__(self, mass=1.0, drag_coefficient=1.0, lift_coefficient=1.0, rotational_drag_coefficient=4.0, rudder_lift_coefficient=0.1, heading=None, position=None):
        self.mass = mass
        self.drag_coefficient = drag_coefficient
        self.lift_coefficient = lift_coefficient
        self.rotational_drag_coefficient = rotational_drag_coefficient
        self.rudder_lift_coefficient = rudder_lift_coefficient
        self.reset_position = position
        self.reset_heading = heading
        self.position = np.array([0.0, 0.0]) if position is None else position
        self.heading = np.array([0.0, 1.0]) if heading is None else heading / np.linalg.norm(heading)
        self.rotational_velocity = 0.0
        self.speed = 0.0

        #hardcoded parameters
        self.rot_inertia = self.mass * 0.5  # Simplified moment of inertia
        self.rudder_dist = 1 # Distance of the rudder relative to boat center


    def calc_wind_acceleration(self, wind_vector):
        relative_wind = wind_vector - self.heading * self.speed
        lift_force = self.lift_coefficient * relative_sail_efficiency(relative_wind, self.heading)

        drag_force = self.drag_coefficient * self.speed**2

        net_force = lift_force - drag_force
        return net_force / self.mass

    def calc_rudder_effect(self, rudder_angle):
        rot_acc = -0.5*self.rudder_lift_coefficient*self.rudder_dist*np.sin(2*rudder_angle)*self.speed / self.rot_inertia
        rot_drag = -0.5*self.rotational_drag_coefficient * (np.sign(self.rotational_velocity)*self.rotational_velocity**2)
        drag = self.rudder_lift_coefficient * (np.sin(rudder_angle)**2) * self.speed

        return rot_acc+rot_drag, drag

    def sail_angle(self, wind_vector):
        relative_wind = wind_vector - self.heading * self.speed

        wind_angle = np.arctan2(relative_wind[1], relative_wind[0])
        boat_angle = np.arctan2(-self.heading[1], -self.heading[0])
        angle_diff = wind_angle - boat_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        optimal_sail_angle = angle_diff / 2

        return optimal_sail_angle

    def update(self, wind_vector, rudder_angle, time_step):
        acceleration = self.calc_wind_acceleration(wind_vector)
        rot_acc, rudder_drag = self.calc_rudder_effect(rudder_angle)
        acceleration -= rudder_drag / self.mass

        self.rotational_velocity += (rot_acc - self.rotational_drag_coefficient*np.sign(self.rotational_velocity)*self.rotational_velocity**2) * time_step
        self.heading = np.array([
            np.cos(self.heading_angle() + self.rotational_velocity * time_step),
            np.sin(self.heading_angle() + self.rotational_velocity * time_step)
        ])

        self.speed += acceleration * time_step
        self.position += self.heading * self.speed * time_step

    def set_heading(self, angle_radians):
        self.heading = np.array([np.cos(angle_radians), np.sin(angle_radians)])

    def heading_angle(self):
        return np.arctan2(self.heading[1], self.heading[0])

    def state(self):
        return self.position, self.heading, self.speed, self.rotational_velocity

    def reset(self):
        self.position = np.array([0.0, 0.0]) if self.reset_position is None else self.reset_position
        self.heading = np.array([0.0, 1.0]) if self.reset_heading is None else self.reset_heading / np.linalg.norm(self.reset_heading)
        self.rotational_velocity = 0.0
        self.speed = 0.0