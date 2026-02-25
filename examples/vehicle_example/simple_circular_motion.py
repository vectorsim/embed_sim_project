# ==============================================================================
# Vehicle Circular Motion Simulation using FMU & VectorSim
# ==============================================================================
# Author   : Paul Abraham
# Date     : 2026-02-24
# Version  : 2.0
#
# Description:
#   This script simulates and visualizes the motion of a kinematic bicycle vehicle
#   performing circular motion using a Modelica FMU and the VectorSim simulation
#   framework. It provides:
#
#       1. FMU-based vehicle model integration:
#           - VehicleKinematicFMU: loads a Modelica FMU for vehicle kinematics
#           - VehiclePathGenerator: generates reference steering and speed signals
#           - VectorSim: simulates signal flow and FMU outputs
#
#       2. Vehicle representation and plotting:
#           - Vehicle class: draws a rectangular vehicle with heading arrow
#           - VehiclePlot class: plots vehicle trajectory with heading arrows,
#             optional grid, and path trail
#
#       3. Modular simulation workflow:
#           - setup_simulation(): initializes FMU, path generator, and simulation
#           - run_simulation(): executes simulation and returns x, y trajectories
#
#       4. Output:
#           - Visual plot of vehicle trajectory with arrows and optional grid
#           - Saves plot as PNG (vehicle_circle_motion.png)
#
# Dependencies:
#   - Python packages: numpy, matplotlib, pathlib
#   - VectorSim framework: sim_core (VectorSim, ODESolver, VectorEnd)
#   - vehicle_fmu module: VehicleKinematicFMU, VehiclePathGenerator
#   - Modelica FMU file: VehicleKinematicBicycle.fmu
#
# Notes:
#   - Vehicle heading is tangent to the trajectory, calculated from FMU outputs.
#   - Last vehicle position is highlighted in red for clarity.
#   - Modular functions allow easy adjustment of FMU path, cruise speed, steering
#     angle, and simulation duration.
# ==============================================================================


import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver
from vehicle_fmu import VehicleKinematicFMU, VehiclePathGenerator


# ------------------------------
# Vehicle Class
# ------------------------------
class Vehicle:
    """Rectangular vehicle with heading arrow."""
    def __init__(self, x=0.0, y=0.0, theta=0.0, length=2.0, width=1.0):
        self.x, self.y, self.theta = x, y, theta
        self.length, self.width = length, width

    def set_pose(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta

    def draw(self, ax, color='blue', alpha=1.0):
        # Rectangle
        rect = Rectangle((-self.length/2, -self.width/2),
                         self.length, self.width,
                         facecolor=color, edgecolor='black', alpha=alpha)
        trans = Affine2D().rotate(self.theta).translate(self.x, self.y) + ax.transData
        rect.set_transform(trans)
        ax.add_patch(rect)
        # Heading arrow
        arrow_length = self.length * 0.9
        ax.arrow(self.x, self.y,
                 arrow_length*np.cos(self.theta),
                 arrow_length*np.sin(self.theta),
                 head_width=0.5, head_length=0.7,
                 fc='red', ec='red')
        return rect


# ------------------------------
# VehiclePlot Class
# ------------------------------
class VehiclePlot:
    """Plots a vehicle moving along a path with heading arrows, grid, and trail."""
    def __init__(self, path_x, path_y, vehicle_length=2.0, vehicle_width=1.0,
                 arrow_skip=10, xlim=(-50,50), ylim=(-50,50), grid_spacing=5.0):
        self.path_x, self.path_y = path_x, path_y
        self.num_points = len(path_x)
        self.vehicle = Vehicle(length=vehicle_length, width=vehicle_width)
        self.arrow_skip, self.xlim, self.ylim, self.grid_spacing = arrow_skip, xlim, ylim, grid_spacing

    def plot(self, save_as=None):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_aspect('equal')
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_title("Vehicle Circular Motion")

        # Grid and path
        ax.set_xticks(np.arange(self.xlim[0], self.xlim[1]+1, self.grid_spacing))
        ax.set_yticks(np.arange(self.ylim[0], self.ylim[1]+1, self.grid_spacing))
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.plot(self.path_x, self.path_y, 'g--', linewidth=1)

        # Draw vehicles along path
        for i in range(0, self.num_points, self.arrow_skip):
            dx, dy = self.path_x[(i+1)%self.num_points]-self.path_x[i], self.path_y[(i+1)%self.num_points]-self.path_y[i]
            theta = np.arctan2(dy, dx)
            self.vehicle.set_pose(self.path_x[i], self.path_y[i], theta)
            self.vehicle.draw(ax, color='blue', alpha=0.6)

        # Last vehicle
        dx, dy = self.path_x[-1]-self.path_x[-2], self.path_y[-1]-self.path_y[-2]
        self.vehicle.set_pose(self.path_x[-1], self.path_y[-1], np.arctan2(dy, dx))
        self.vehicle.draw(ax, color='red', alpha=1.0)

        if save_as:
            plt.savefig(save_as, dpi=150, bbox_inches='tight')
            print(f"✅ Saved plot as '{save_as}'")
        plt.show()


# ------------------------------
# Simulation Functions
# ------------------------------
def setup_simulation(fmu_path, delta=0.15, accel=0.25, cruise_speed=0.01, T=28.0, dt=0.75):
    """Initialize FMU vehicle, path generator, and VectorSim."""
    path_input = VehiclePathGenerator('path_input', delta=delta, accel=accel, cruise_speed=cruise_speed, use_c_backend= False,)
    fmu_vehicle = VehicleKinematicFMU('fmu_vehicle', fmu_path=str(fmu_path))
    output = VectorEnd('output')
    path_input >> fmu_vehicle >> output

    sim = EmbedSim(sinks=[output], T=T, dt=dt, solver=ODESolver.EULER)
    sim.scope.add(fmu_vehicle, label="vehicle_state")
    return sim


def run_simulation(sim):
    """Run simulation and return x, y trajectory."""
    sim.print_topology_tree()
    sim.print_topology_sources2sink()
    sim.run()
    x_traj = sim.scope.get_signal("vehicle_state", 0)
    y_traj = sim.scope.get_signal("vehicle_state", 1)
    return x_traj, y_traj


# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).parent
    fmu_path = base_dir / "modelica" / "VehicleKinematicBicycle.fmu"

    sim = setup_simulation(fmu_path)
    x_traj, y_traj = run_simulation(sim)

    vp = VehiclePlot(x_traj, y_traj, vehicle_length=3.0, vehicle_width=1.5,
                     arrow_skip=1, xlim=(-25,25), ylim=(-5,45), grid_spacing=10)
    vp.plot(save_as="vehicle_circle_motion.png")