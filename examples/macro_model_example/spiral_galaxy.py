"""
spiral_galaxy_fmu_example.py
============================

Educational example: SpiralGalaxy FMU (Co-Simulation) with EmbedSim
- Loads SpiralGalaxy.fmu
- Runs simulation for N stars
- Animates x-y positions
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── Add EmbedSim to path ───────────────────────────────────────────────
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from embedsim.core_blocks import VectorBlock, VectorSignal
from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver
from embedsim.fmu_blocks import FMUBlock

# ── Simulation configuration ──────────────────────────────────────────
class SimConfig:
    t_stop = 10.0          # seconds
    dt     = 0.01          # simulation timestep
    solver = ODESolver.RK4  # ODE solver for any internal blocks

    fmu_path = base_dir / "macro_model_example" / "modelica" / "SpiralGalaxy.fmu"

# ── FMU wrapper block ─────────────────────────────────────────────────
class SpiralGalaxyFMU(FMUBlock):
    """FMUBlock for SpiralGalaxy Co-Simulation FMU"""

    def __init__(self, name: str, cfg: SimConfig):
        super().__init__(
            name=name,
            fmu_path=str(cfg.fmu_path),
            input_names=[],        # Co-Simulation, no inputs
            output_names=[         # Expose positions of first 3 stars for simplicity
                "x[1]", "y[1]", "z[1]",
                "x[2]", "y[2]", "z[2]",
                "x[3]", "y[3]", "z[3]",
            ]
        )

    def compute(self, t: float, dt: float, input_values=None):
        result = super().compute(t, dt, input_values)
        self.last_values = np.array([v.value[0] for v in result])
        return result

# ── Main simulation ───────────────────────────────────────────────────
def main(cfg: SimConfig = None):
    cfg = cfg or SimConfig()

    # Instantiate FMU block and sink
    galaxy_fmu = SpiralGalaxyFMU("spiral_galaxy", cfg)
    sink       = VectorEnd("sink")

    # Connect FMU to sink
    galaxy_fmu >> sink

    # Create simulation
    sim = EmbedSim(sinks=[sink], T=cfg.t_stop, dt=cfg.dt, solver=cfg.solver)

    # Register signals
    sim.scope.add(galaxy_fmu, "Galaxy_Positions")

    # Run simulation
    sim.run(progress_bar=True)
    print("Simulation finished")

    # Extract data
    t_arr = np.array(sim.scope.t)
    pos_arr = np.array(sim.scope.get_signal("Galaxy_Positions"))  # shape: (timesteps, outputs)

    # Animate first 3 stars (x, y)
    fig, ax = plt.subplots(figsize=(6,6))
    scat = ax.scatter(pos_arr[0,0:3], pos_arr[0,3:6], s=20, c=['red','green','blue'])

    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title("SpiralGalaxy FMU (first 3 stars)")

    def update(frame):
        scat.set_offsets(np.c_[pos_arr[frame,0:3], pos_arr[frame,3:6]])
        return scat,

    anim = FuncAnimation(fig, update, frames=len(t_arr), interval=20, blit=True)
    anim.save("spiral_galaxy_animation.gif", dpi=120)
    print("Animation saved → spiral_galaxy_animation.gif")

if __name__ == "__main__":
    main()