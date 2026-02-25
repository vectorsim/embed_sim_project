"""
vehicle_fmu_animation_smooth.py
===============================
ControlForge — Smooth FMU Vehicle Animation with Interpolation

Purpose:
    Run FMU vehicle simulation using VectorSim, then animate smoothly.
    Uses linear interpolation between FMU outputs for higher FPS.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.transforms import Affine2D
from pathlib import Path
import time

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver
from vehicle_fmu import VehicleKinematicFMU, VehiclePathGenerator

# ------------------------------
# VEHICLE CLASS
# ------------------------------
class Vehicle:
    """Rectangular vehicle with heading arrow."""
    def __init__(self, length=2.0, width=1.0):
        self.x, self.y, self.theta = 0, 0, 0
        self.length, self.width = length, width
        self.rect_patch = None
        self.arrow_patch = None

    def set_pose(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta

    def draw(self, ax, color='blue', alpha=1.0):
        """Draw or update vehicle rectangle and heading arrow."""
        if self.rect_patch is None:
            self.rect_patch = Rectangle((-self.length/2, -self.width/2),
                                        self.length, self.width,
                                        facecolor=color, edgecolor='black', alpha=alpha)
            trans = Affine2D().rotate(self.theta).translate(self.x, self.y) + ax.transData
            self.rect_patch.set_transform(trans)
            ax.add_patch(self.rect_patch)

            arrow_length = self.length * 0.9
            dx = arrow_length * np.cos(self.theta)
            dy = arrow_length * np.sin(self.theta)
            self.arrow_patch = FancyArrowPatch((self.x, self.y), (self.x+dx, self.y+dy),
                                               color='red', arrowstyle='-|>', mutation_scale=15)
            ax.add_patch(self.arrow_patch)
        else:
            trans = Affine2D().rotate(self.theta).translate(self.x, self.y) + ax.transData
            self.rect_patch.set_transform(trans)

            arrow_length = self.length * 0.9
            dx = arrow_length * np.cos(self.theta)
            dy = arrow_length * np.sin(self.theta)
            self.arrow_patch.set_positions((self.x, self.y), (self.x+dx, self.y+dy))

# ------------------------------
# SMOOTH ANIMATION FUNCTION
# ------------------------------
def animate_vehicle_smooth(x_traj, y_traj, theta_traj,
                           vehicle_length=2.0, vehicle_width=1.0,
                           xlim=(-50,50), ylim=(-50,50),
                           fps=60):
    """
    Animate vehicle using interpolated trajectory for smooth motion.
    """
    # Original simulation time steps
    num_points = len(x_traj)
    t_orig = np.linspace(0, 1, num_points)

    # Interpolated time for smooth FPS
    t_smooth = np.linspace(0, 1, int(num_points * fps * (1/num_points)))

    # Interpolate trajectories
    x_smooth = np.interp(t_smooth, t_orig, x_traj)
    y_smooth = np.interp(t_smooth, t_orig, y_traj)
    theta_smooth = np.interp(t_smooth, t_orig, theta_traj)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("FMU Vehicle Smooth Animation")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Draw full path trail (faint)
    ax.plot(x_traj, y_traj, 'g--', linewidth=1, alpha=0.3)

    vehicle = Vehicle(length=vehicle_length, width=vehicle_width)

    trail_x, trail_y = [], []

    dt = 1/fps  # time per frame
    for i in range(len(x_smooth)):
        vehicle.set_pose(x_smooth[i], y_smooth[i], theta_smooth[i])
        vehicle.draw(ax, color='blue', alpha=0.7)

        trail_x.append(x_smooth[i])
        trail_y.append(y_smooth[i])
        ax.plot(trail_x, trail_y, 'g--', linewidth=1)

        plt.pause(dt)

    plt.show()

# ------------------------------
# MAIN SCRIPT
# ------------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).parent
    fmu_path = base_dir / "modelica" / "VehicleKinematicBicycle.fmu"

    # --- Blocks ---
    path_input = VehiclePathGenerator('path_input', delta=0.15, accel=0.25, cruise_speed=0.01)
    fmu_vehicle = VehicleKinematicFMU('fmu_vehicle', fmu_path=str(fmu_path))
    output = VectorEnd('output')

    path_input >> fmu_vehicle >> output

    # --- Simulation ---
    T_SIM = 35.0
    DT = 0.15

    sim = EmbedSim(sinks=[output], T=T_SIM, dt=DT, solver=ODESolver.EULER)
    sim.scope.add(fmu_vehicle, label="vehicle_state")

    sim.print_topology_tree()
    sim.print_topology_sources2sink()

    print("\nRunning simulation...")
    sim.run()

    # Extract FMU trajectory
    x_traj = sim.scope.get_signal("vehicle_state", 0)
    y_traj = sim.scope.get_signal("vehicle_state", 1)
    theta_traj = sim.scope.get_signal("vehicle_state", 2)

    # Animate smoothly at 60 FPS
    animate_vehicle_smooth(x_traj, y_traj, theta_traj,
                           vehicle_length=3.0, vehicle_width=1.5,
                           xlim=(-25,25), ylim=(-5,45),
                           fps=60)