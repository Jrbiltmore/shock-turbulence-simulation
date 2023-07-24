# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from fluid_solver import FluidSolver
from turbulence_model import TurbulenceModel
from shock_model import ShockModel
from visualization import visualize_results
from analysis import analyze_turbulence_statistics
from input_parameters import InputParameters
from utils import save_simulation_settings, save_simulation_results

def main():
    # Load input parameters from a configuration file or set them manually
    input_params = InputParameters.from_file("input.yaml")  # Example: Load from a YAML file
    save_simulation_settings("simulation_settings.yaml", input_params)  # Save settings for reference

    # Extract input parameters from the InputParameters object
    grid_size = input_params.grid_size
    time_step = input_params.time_step
    num_time_steps = input_params.num_time_steps
    turbulence_model = TurbulenceModel(input_params.turbulence_model_params)
    shock_model = ShockModel(input_params.shock_model_params)

    # Create the Fluid Solver instance
    fluid_solver = FluidSolver(grid_size, time_step, num_time_steps, turbulence_model, shock_model)

    # Run the simulation
    fluid_solver.run_simulation()

    # Get the simulation results
    velocity, pressure, density = fluid_solver.get_results()

    # Visualize the simulation results
    visualize_results(velocity, pressure, density)

    # Analyze turbulence statistics
    analyze_turbulence_statistics(velocity)

    # Save the final simulation results
    save_simulation_results("simulation_results", fluid_solver)

if __name__ == "__main__":
    main()
