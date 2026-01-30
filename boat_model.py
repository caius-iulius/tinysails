import numpy as np
from numpy import dot

def relative_sail_efficiency(wind_vector, boat_versor):
    efficiency = 0.5 * (np.linalg.norm(wind_vector) + dot(wind_vector, boat_versor))
    return efficiency

class Boat:
    def __init__(self, mass=1.0, drag_coefficient=1.0, lift_coefficient=1.0, rudder_lift_coefficient=0.1):
        self.mass = mass
        self.drag_coefficient = drag_coefficient
        self.lift_coefficient = lift_coefficient
        self.rudder_lift_coefficient = rudder_lift_coefficient
        self.position = np.array([0.0, 0.0])
        self.heading = np.array([0.0, 1.0])
        self.rotational_velocity = 0.0
        self.speed = 0.0

    def calc_wind_acceleration(self, wind_vector):
        relative_wind = wind_vector - self.heading * self.speed
        lift_force = self.lift_coefficient * relative_sail_efficiency(relative_wind, self.heading)
        drag_force = self.drag_coefficient * self.speed**2

        net_force = lift_force - drag_force
        return net_force / self.mass

    def calc_rudder_effect(self, rudder_angle):
        rot_inertia = self.mass * 0.5  # Simplified moment of inertia
        rudder_dist = 1 # Distance of the rudder relative to boat center

        rot_acc = -0.5*self.rudder_lift_coefficient*rudder_dist*np.sin(2*rudder_angle)*self.speed / rot_inertia
        drag = self.rudder_lift_coefficient * (np.sin(rudder_angle)**2) * self.speed

        # Placeholder for rudder effect calculation
        return rot_acc, drag

    def sail_angle(self, wind_vector):
        relative_wind = wind_vector - self.heading * self.speed

        wind_angle = np.arctan2(relative_wind[1], relative_wind[0])
        boat_angle = np.arctan2(-self.heading[1], -self.heading[0])
        angle_diff = wind_angle - boat_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        optimal_sail_angle = angle_diff / 2

        return optimal_sail_angle

    def update(self, wind_vector, rudder_angle, time_step):
        rot_drag = 4 # rotational drag coefficient
        acceleration = self.calc_wind_acceleration(wind_vector)
        rot_acc, rudder_drag = self.calc_rudder_effect(rudder_angle)
        acceleration -= rudder_drag / self.mass

        self.rotational_velocity += (rot_acc - rot_drag*np.sign(self.rotational_velocity)*self.rotational_velocity**2) * time_step
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
