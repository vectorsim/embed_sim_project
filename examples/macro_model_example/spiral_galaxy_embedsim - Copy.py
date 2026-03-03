"""
spiral_galaxy_live_500.py
=========================

Live slow-motion animation of a spiral galaxy simulated via FMU
using the EmbedSim framework, matching Modelica SpiralGalaxy with 500 stars.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# EmbedSim Bootstrap
# ─────────────────────────────────────────────
def add_parent_to_syspath(levels=2):
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(base.parents[levels - 1]))

add_parent_to_syspath(2)

from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver
from embedsim.fmu_blocks import FMUBlock

print("✅ EmbedSim framework loaded")


# ─────────────────────────────────────────────
# Simulation Parameters
# ─────────────────────────────────────────────
N_STARS = 500      # matches Modelica SpiralGalaxy
T_MYR   = 800.0
DT_MYR  = 1.5

# Conversion: kpc·km/s → Myr
KPC_KMS_TO_MYR = 977.79
T_NAT  = T_MYR / KPC_KMS_TO_MYR
DT_NAT = DT_MYR / KPC_KMS_TO_MYR


# ─────────────────────────────────────────────
# Generate Spiral Galaxy Initial Conditions
# ─────────────────────────────────────────────
def generate_spiral_initial_conditions(n_stars: int, n_arms: int = 4, pitch: float = 0.28, r_min: float = 3.0, dr: float = 0.15):
    r_init = r_min + dr * np.arange(n_stars)
    arm_off = 2 * np.pi * np.arange(n_stars) / n_stars * n_arms
    theta = pitch * r_init + arm_off
    omega_c = np.sqrt(4.302e-6 * 1e11 / (r_init**2 + (6.0 + 0.5)**2)**1.5)

    x = r_init * np.cos(theta + 0.02 * (np.arange(n_stars)/n_stars - 0.5))
    y = r_init * np.sin(theta + 0.02 * (np.arange(n_stars)/n_stars - 0.5))
    z = 0.07 * (np.arange(n_stars) - n_stars/2)/n_stars + 0.01*(np.arange(n_stars)/n_stars - 0.5)

    vx = -np.sin(theta) * r_init * omega_c
    vy =  np.cos(theta) * r_init * omega_c
    vz = np.zeros(n_stars)

    init_states = {f"{var}[{i+1}]": val for i, (val, var) in enumerate(zip(
        np.stack([x, y, z, vx, vy, vz], axis=1).T.flatten(),
        ["x", "y", "z", "vx", "vy", "vz"] * n_stars
    ))}

    star_arm = np.arange(n_stars) % n_arms

    return init_states, star_arm


# ─────────────────────────────────────────────
# Build FMU Block
# ─────────────────────────────────────────────
def build_galaxy_fmu(fmu_path: Path, n_stars: int, init_states: dict) -> FMUBlock:
    output_names = [f"{var}[{i+1}]" for i in range(n_stars) for var in ["x","y","z","vx","vy","vz"]]
    return FMUBlock(
        name="galaxy_fmu",
        fmu_path=str(fmu_path),
        input_names=[],
        output_names=output_names,
        parameters={
            "N": n_stars,
            "G": 4.302e-6,
            "Md": 1e11,
            "a": 6.0,
            "b": 0.5,
            **init_states
        }
    )


# ─────────────────────────────────────────────
# Live Animation
# ─────────────────────────────────────────────
def animate_galaxy(XH: np.ndarray, YH: np.ndarray, t_arr_Myr: np.ndarray, star_arm: np.ndarray, frame_delay: float = 0.2):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("black")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")

    colors = plt.cm.viridis(star_arm / star_arm.max())
    scatter = ax.scatter(XH[0], YH[0], s=8, c=colors, edgecolors="white", linewidths=0.2)

    plt.ion()
    plt.show()

    for frame in range(len(XH)):
        scatter.set_offsets(np.c_[XH[frame], YH[frame]])
        ax.set_title(f"Time = {t_arr_Myr[frame]:.1f} Myr", color="white")
        plt.pause(frame_delay)

    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    base_dir = Path(__file__).parent
    fmu_path = base_dir / "modelica" / "SpiralGalaxy.fmu"  # path to your FMU

    init_states, star_arm = generate_spiral_initial_conditions(N_STARS)
    galaxy_fmu = build_galaxy_fmu(fmu_path, N_STARS, init_states)

    trajectory = VectorEnd("trajectory")
    galaxy_fmu >> trajectory

    sim = EmbedSim(
        sinks=[trajectory],
        T=T_NAT,
        dt=DT_NAT,
        solver=ODESolver.EULER
    )
    sim.scope.add(galaxy_fmu, label="galaxy_state", record_full=True)

    print("🚀 Running FMU simulation...")
    sim.run(verbose=True, progress_bar=True)

    hist = sim.scope.get_full_signal("galaxy_state")
    XH = hist[:, 0::6]
    YH = hist[:, 1::6]
    t_arr_Myr = np.array(sim.scope.t) * KPC_KMS_TO_MYR

    print(f"✅ Simulation finished | hist.shape={hist.shape}")
    animate_galaxy(XH, YH, t_arr_Myr, star_arm)


if __name__ == "__main__":
    main()