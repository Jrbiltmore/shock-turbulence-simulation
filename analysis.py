# analysis.py

import numpy as np
import matplotlib.pyplot as plt

def compute_average_density(density_data):
    # Compute the average density from the given density data (time series)
    return np.mean(density_data, axis=0)

def compute_average_temperature(temperature_data):
    # Compute the average temperature from the given temperature data (time series)
    return np.mean(temperature_data, axis=0)

def compute_temperature_fluctuations(temperature_data):
    # Compute the fluctuations in temperature from the given temperature data (time series)
    average_temperature = compute_average_temperature(temperature_data)
    return temperature_data - average_temperature

def plot_average_density(density_data):
    # Plot the time-averaged plasma density
    average_density = compute_average_density(density_data)
    plt.figure(figsize=(8, 6))
    plt.imshow(average_density, cmap='hot', origin='lower')
    plt.title('Average Plasma Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.show()

def plot_average_temperature(temperature_data):
    # Plot the time-averaged plasma temperature
    average_temperature = compute_average_temperature(temperature_data)
    plt.figure(figsize=(8, 6))
    plt.imshow(average_temperature, cmap='coolwarm', origin='lower')
    plt.title('Average Plasma Temperature')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.show()

def plot_temperature_fluctuations(temperature_data):
    # Plot the fluctuations in plasma temperature
    temperature_fluctuations = compute_temperature_fluctuations(temperature_data)
    plt.figure(figsize=(8, 6))
    plt.imshow(temperature_fluctuations, cmap='coolwarm', origin='lower')
    plt.title('Plasma Temperature Fluctuations')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.show()

def plot_density_and_temperature(density_data, temperature_data):
    # Plot plasma density and temperature as subplots
    average_density = compute_average_density(density_data)
    average_temperature = compute_average_temperature(temperature_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot plasma density
    axes[0].imshow(average_density, cmap='hot', origin='lower')
    axes[0].set_title('Average Plasma Density')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].colorbar()

    # Plot plasma temperature
    axes[1].imshow(average_temperature, cmap='coolwarm', origin='lower')
    axes[1].set_title('Average Plasma Temperature')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].colorbar()

    plt.tight_layout()
    plt.show()

def plot_3d_density_surface(density_data, time_steps):
    # Plot 3D surface of plasma density over time
    x = np.arange(density_data.shape[1])
    y = np.arange(density_data.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for t in range(time_steps):
        Z = density_data[t]
        ax.plot_surface(X, Y, Z, cmap='hot', alpha=0.5)

    ax.set_title('3D Plasma Density Surface')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')
    plt.show()

# Additional analysis functions can be added here based on your specific analysis needs.
# For example, computing plasma flow velocities, energy distribution, particle trajectories, etc.
