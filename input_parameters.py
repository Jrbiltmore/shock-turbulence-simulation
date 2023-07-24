# input_parameters.py

class InputParameters:
    def __init__(self):
        # Simulation parameters
        self.grid_size = 100  # Size of the simulation grid
        self.time_step = 0.01  # Time step for the simulation
        self.num_time_steps = 100  # Number of time steps to run the simulation

        # Turbulence model parameters
        self.turbulence_model = "Smagorinsky"  # Turbulence model to use (e.g., "Smagorinsky", "k-epsilon", etc.)
        self.Cs = 0.1  # Smagorinsky constant (tunable parameter)

        # Shock model parameters
        self.shock_model = "TVD"  # Shock model to use (e.g., "TVD", "Roe", etc.)
        self.C_viscosity = 0.1  # Artificial viscosity coefficient (tunable parameter)

        # Plasma physics parameters
        self.electron_density = 1e18  # Electron density of the plasma (in m^-3)
        self.ion_temperature = 1.0  # Ion temperature of the plasma (in eV)

        # Add more parameters as needed for your specific simulation and analysis

    def get_parameters(self):
        # Return the input parameters as a dictionary
        parameters = {
            "grid_size": self.grid_size,
            "time_step": self.time_step,
            "num_time_steps": self.num_time_steps,
            "turbulence_model": self.turbulence_model,
            "Cs": self.Cs,
            "shock_model": self.shock_model,
            "C_viscosity": self.C_viscosity,
            "electron_density": self.electron_density,
            "ion_temperature": self.ion_temperature,
            # Add more parameters as needed
        }
        return parameters

# Create an instance of the InputParameters class to access the input parameters
input_parameters = InputParameters()
