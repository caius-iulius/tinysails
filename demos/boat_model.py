import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from boat_model import Boat

def boat_equilibrium_speed(wind_vector, boat):
    def equilibrium_equation(boat_speed):
        boat.speed = boat_speed
        return boat.calc_wind_acceleration(wind_vector)

    boat_speed_solution = fsolve(equilibrium_equation, 0.0)[0]

    return boat_speed_solution

angles = np.linspace(0, 2 * np.pi, 360)
wind_speeds = np.linspace(2.0, 16.0, 8)

plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection='polar')
ax.set_theta_zero_location('N') #type: ignore

for wind_speed in wind_speeds:
    efficiencies = []
    for angle in angles:
        boat = Boat(drag_coefficient=1.0, lift_coefficient=45.0)
        boat.set_heading(angle)
        efficiency = boat_equilibrium_speed(np.array([-wind_speed,0]), boat)

        efficiencies.append(efficiency)
    ax.plot(angles, efficiencies, label=f'wind={wind_speed:.1f}')

ax.set_title("Boat Equilibrium Speed vs Boat Heading\n(varying wind speeds)")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()