# utils.py

import numpy as np

def normalize_vector(vector):
    # Normalize a 2D vector (array) to have unit magnitude
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector
    return vector / magnitude

def calculate_temperature(velocity_magnitude, molar_mass, boltzmann_constant):
    # Calculate the temperature of the plasma based on the velocity magnitude and molar mass
    temperature = (molar_mass * velocity_magnitude**2) / (2 * boltzmann_constant)
    return temperature

def calculate_mach_number(velocity, speed_of_sound):
    # Calculate the Mach number of the flow
    magnitude = np.linalg.norm(velocity)
    if speed_of_sound == 0:
        return 0
    return magnitude / speed_of_sound

def compute_density_from_temperature_and_pressure(temperature, pressure, molar_mass, boltzmann_constant):
    # Compute the density of the plasma using the ideal gas law
    density = pressure / (boltzmann_constant * temperature)
    density /= molar_mass
    return density

def interpolate_linear(x, x_values, y_values):
    # Perform linear interpolation given x values, y values, and the desired x coordinate
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    if x <= x_values[0]:
        return y_values[0]
    if x >= x_values[-1]:
        return y_values[-1]

    left_index = np.searchsorted(x_values, x) - 1
    right_index = left_index + 1

    x_left, x_right = x_values[left_index], x_values[right_index]
    y_left, y_right = y_values[left_index], y_values[right_index]

    slope = (y_right - y_left) / (x_right - x_left)
    y = y_left + slope * (x - x_left)
    return y

def plot_data(x_values, y_values, xlabel, ylabel, title):
    # Plot data using Matplotlib
    import matplotlib.pyplot as plt
    plt.plot(x_values, y_values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

# Add more utility functions and helper methods as needed for your specific simulation and analysis needs.
