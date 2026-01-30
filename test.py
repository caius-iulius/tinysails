import numpy as np
from numpy import dot
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def relative_sail_efficiency(wind_vector, boat_versor):
    efficiency = 0.5 * (np.linalg.norm(wind_vector) + dot(wind_vector, boat_versor))
    return efficiency

def absolute_sail_efficiency(wind_vector, boat_heading, drag_coefficient=1.0, lift_coefficient=1.0):
    # find boat_speed such that:
    # lift_coefficient * relative_sail_efficiency(wind_vector - boat_heading*boat_speed, boat_heading) = drag_coefficient * boat_speed

    def equilibrium_equation(boat_speed):
        relative_wind = wind_vector - boat_heading * boat_speed
        lift_force = lift_coefficient * relative_sail_efficiency(relative_wind, boat_heading)
        drag_force = drag_coefficient * boat_speed**2
        return lift_force - drag_force

    boat_speed_solution = fsolve(equilibrium_equation, 0.0)[0]

    return boat_speed_solution

if __name__ == "__main__":
    wind_vector =  np.array([-1.0, 0.0])

    #draw polar plot
    angles = np.linspace(0, 2 * np.pi, 360)
    drag_coefficients = np.linspace(0.1, 1.0, 10)

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')
    #set the 0 to the top
    ax.set_theta_zero_location('N')

    for drag_coeff in drag_coefficients:
        efficiencies = []
        for angle in angles:
            boat_versor = np.array([np.cos(angle), np.sin(angle)])
            #efficiency = absolute_sail_efficiency(wind_vector, boat_versor, drag_coeff)
            efficiency = absolute_sail_efficiency(np.array([-drag_coeff,0]), boat_versor, 0.2, 0.5)
            efficiencies.append(efficiency)
        ax.plot(angles, efficiencies, label=f'drag={drag_coeff:.1f}')

    ax.set_title("Absolute Sail Efficiency vs Boat Heading\n(varying drag coefficients)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()