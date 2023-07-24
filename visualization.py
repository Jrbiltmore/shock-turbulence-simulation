# visualization.py
import numpy as np
import matplotlib.pyplot as plt

def visualize_density(density):
    # Visualize plasma density using a contour plot
    plt.figure(figsize=(8, 6))
    plt.contourf(density, cmap='plasma')
    plt.title('Plasma Density Contour')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Density')
    plt.grid(True)
    plt.show()

def visualize_temperature(temperature):
    # Visualize plasma temperature using a contour plot
    plt.figure(figsize=(8, 6))
    plt.contourf(temperature, cmap='hot')
    plt.title('Plasma Temperature Contour')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Temperature')
    plt.grid(True)
    plt.show()

def visualize_plasma_properties(density, temperature):
    # Visualize plasma properties (density and temperature) using subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot plasma density
    axes[0].contourf(density, cmap='plasma')
    axes[0].set_title('Plasma Density Contour')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True)
    axes[0].colorbar(label='Density')

    # Plot plasma temperature
    axes[1].contourf(temperature, cmap='hot')
    axes[1].set_title('Plasma Temperature Contour')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True)
    axes[1].colorbar(label='Temperature')

    plt.tight_layout()
    plt.show()

def visualize_plasma_properties_3d(density, temperature):
    # Visualize plasma properties (density and temperature) using a 3D plot
    x, y = np.meshgrid(np.arange(density.shape[1]), np.arange(density.shape[0]))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot plasma density
    ax.plot_surface(x, y, density, cmap='plasma', rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_title('Plasma Density 3D Surface')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')

    # Plot plasma temperature
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, temperature, cmap='hot', rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_title('Plasma Temperature 3D Surface')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature')

    plt.show()

# You can add more visualization functions as needed based on the specific properties and phenomena you want to visualize in the plasma simulation.

