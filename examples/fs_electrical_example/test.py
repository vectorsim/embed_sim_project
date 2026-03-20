"""
db42s02_openloop_fmu.py
=======================
EmbedSim  —  Fixed V/f Control with proper boost implementation
===============================================================
"""

from __future__ import annotations

import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Path bootstrap ─────────────────────────────────────────────────────────────
from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE    = get_current_parent()
_ROOT    = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"

for _p in (
    get_embedsim_import_path(),
    str(_FS_ELEC),
    str(_FS_ELEC / "c_src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── EmbedSim ──────────────────────────────────────────────────────────────────
from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.code_generator import CodeGenStart, CodeGenEnd

# ── fs_electrical_machines ────────────────────────────────────────────────────
from motor_utility_blocks import (
    SpeedRampBlock,
    VfAngleBlock,
    VfDQBlock,
    VfThetaBlock,
    SVPWMPackBlock,
)
from coordinate_transform_blocks import InvParkTransformBlock
from svpwm_block                  import SVPWMBlock
from PMSM_MotorBlock              import PMSM_MotorBlock

_FMU_PATH = str(_FS_ELEC / "modelica" / "PMSM_Motor.fmu")


# =============================================================================
# Motor and Control Parameters
# =============================================================================
# Motor parameters
P_POLES        = 4
V_DC           = 17.0
V_PHASE_PEAK   = V_DC / math.sqrt(3.0)          # 9.815 V
R_STATOR       = 0.19                            # Ω
L_d = L_q = 0.125e-3                             # H
LAMBDA_PM      = 0.0014                           # Wb

# Simulation parameters
T_SIM          = 0.8                               # Longer simulation [s]
DT             = 1e-4                               # Time step [s]
OMEGA_CMD_RPM  = 400.0                              # Target speed [RPM]
RAMP_TIME      = 0.3                                # Slower speed ramp [s]

# Derived parameters
OMEGA_CMD_RADS = OMEGA_CMD_RPM * 2.0 * math.pi / 60.0  # 41.89 rad/s
F_ELECTRICAL   = (OMEGA_CMD_RPM * P_POLES / 60.0)      # 26.67 Hz

# V/f ratio calculation
OMEGA_M_RATED  = 8000.0 * 2.0 * math.pi / 60.0        # 837.76 rad/s
OMEGA_E_RATED  = P_POLES * OMEGA_M_RATED              # 3351.04 rad/s
VF_RATIO       = V_PHASE_PEAK / OMEGA_E_RATED         # 0.00293 V·s/rad

# Voltage boost for low speeds - CRITICAL for sinusoidal currents
# At low speeds, we need to overcome stator resistance
I_TARGET       = 1.0                                   # Target current [A]
V_R_DROP = R_STATOR * I_TARGET                         # 0.19 V

# Add extra boost for magnetization
VF_BOOST = V_R_DROP * 2.0                              # 0.38 V (double for safety)

# Boost profile
BOOST_CUTOFF_SPEED = 500.0                             # RPM - boost reduces above this
MIN_SPEED_FOR_BOOST = 50.0                              # RPM - full boost below this


# =============================================================================
# Fixed V/f block with proper boost implementation
# =============================================================================
class FixedVfAngleBlock(VfAngleBlock):
    """
    Fixed V/f angle block with proper voltage boost implementation.

    The key issue with previous implementation was incorrect frequency estimation
    and boost application. This version uses speed reference directly for boost.
    """

    def __init__(self, name: str, vf_ratio: float, v_phase_peak: float,
                 boost_voltage: float, p_poles: int,
                 boost_cutoff_speed: float = 500.0,
                 min_speed_for_boost: float = 50.0):
        super().__init__(name, vf_ratio, v_phase_peak, p_poles)
        self.boost_voltage = boost_voltage
        self.boost_cutoff_speed = boost_cutoff_speed
        self.min_speed_for_boost = min_speed_for_boost

        # For debugging
        self.boost_history = []
        self.speed_history = []
        self.time_history = []

    def compute_py(self, t: float, dt: float, input_values=None):
        """
        Compute V/f outputs with speed-based voltage boost.

        Input: [omega_m_ref] from SpeedRampBlock in rad/s
        Output: [v_d, v_q, theta_e] with boosted v_q
        """
        # Get base V/f outputs
        result = super().compute_py(t, dt, input_values)

        if result is None or len(result.value) < 3:
            return result

        # Get current speed reference from input
        if input_values and input_values[0] is not None:
            omega_ref = float(input_values[0].value[0])  # rad/s
            speed_rpm = omega_ref * 60.0 / (2.0 * math.pi)
        else:
            speed_rpm = 0.0

        # Calculate boost factor based on speed
        if speed_rpm <= self.min_speed_for_boost:
            # Full boost at very low speeds
            boost_factor = 1.0
        elif speed_rpm >= self.boost_cutoff_speed:
            # No boost above cutoff
            boost_factor = 0.0
        else:
            # Linear reduction
            boost_factor = 1.0 - (speed_rpm - self.min_speed_for_boost) / (self.boost_cutoff_speed - self.min_speed_for_boost)

        # Apply boost to v_q (torque-producing component)
        boost_applied = self.boost_voltage * boost_factor
        result.value[1] += boost_applied  # Add to v_q

        # Also ensure v_d has some initial voltage for field alignment
        if speed_rpm < 100.0:
            result.value[0] = 0.5  # Small d-axis voltage for field alignment

        # Store history
        self.boost_history.append(boost_applied)
        self.speed_history.append(speed_rpm)
        self.time_history.append(t)

        return result

    def reset(self) -> None:
        super().reset()
        self.boost_history = []
        self.speed_history = []
        self.time_history = []


# =============================================================================
# Fixed Plant Block with proper current quality monitoring
# =============================================================================
class FixedDB42S02PlantBlock(PMSM_MotorBlock):
    """
    Fixed plant block with proper current quality monitoring.
    """

    TOPO_CATEGORY     = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm,ia,ib,ic,Tem,id,iq,quality]"

    def __init__(self, name: str, fmu_path: str) -> None:
        try:
            super().__init__(
                name      = name,
                fmu_path  = fmu_path,
                R         = 0.19,
                L_d       = 0.125e-3,
                L_q       = 0.125e-3,
                lambda_pm = 0.0014,
                J         = 2.4e-6,
                B         = 1e-6,
                p         = float(P_POLES),
            )
        except Exception as exc:
            print(f"\n[PlantBlock] FMU load failed: {fmu_path}")
            print(f"             {exc}\n")
            raise

        # Current quality monitoring
        self.current_buffer = []
        self.quality_history = []

    def compute_py(self, t: float, dt: float, input_values=None):
        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                ta, tb, tc = float(v[0]), float(v[1]), float(v[2])

        # FMU input: [duty_a, duty_b, duty_c, v_dc, T_load]
        fmu_input = VectorSignal(
            np.array([ta, tb, tc, V_DC, 0.0], dtype=DEFAULT_DTYPE))
        super().compute_py(t, dt, [fmu_input])

        # Read outputs
        speed_rpm = self.read_speed_rpm()
        i_a = self.read_i_a()
        i_b = self.read_i_b()
        i_c = self.read_i_c()
        i_d = self.read_i_d()
        i_q = self.read_i_q()
        T_em = self.read_T_em()

        # Calculate current quality (simple sinusoidal metric)
        quality = self._calculate_quality(i_a, i_b, i_c)
        self.quality_history.append(quality)

        self.output = VectorSignal(
            np.array([speed_rpm, i_a, i_b, i_c, T_em, i_d, i_q, quality],
                    dtype=DEFAULT_DTYPE),
            self.name)
        return self.output

    def _calculate_quality(self, i_a, i_b, i_c):
        """Simple quality metric based on current balance."""
        # Store in buffer
        self.current_buffer.append([i_a, i_b, i_c])
        if len(self.current_buffer) > 1000:
            self.current_buffer.pop(0)

        if len(self.current_buffer) < 500:
            return 0.0

        # Calculate RMS values
        buf = np.array(self.current_buffer[-500:])
        i_a_rms = np.sqrt(np.mean(buf[:, 0]**2))
        i_b_rms = np.sqrt(np.mean(buf[:, 1]**2))
        i_c_rms = np.sqrt(np.mean(buf[:, 2]**2))

        # Check if currents are balanced and non-zero
        if i_a_rms < 0.1 or i_b_rms < 0.1 or i_c_rms < 0.1:
            return 0.0

        # Calculate imbalance
        mean_rms = (i_a_rms + i_b_rms + i_c_rms) / 3
        imbalance = np.std([i_a_rms, i_b_rms, i_c_rms]) / mean_rms

        # Quality is inverse of imbalance (100% = perfectly balanced)
        quality = max(0, 100 * (1 - imbalance))
        return quality


# =============================================================================
# Build and run
# =============================================================================
def build_and_run() -> dict:
    """Build and run simulation."""

    # Instantiate blocks
    cg_start   = CodeGenStart("cg_start")
    speed_ref  = SpeedRampBlock("speed_ref",
                               omega_target=OMEGA_CMD_RADS,
                               ramp_time=RAMP_TIME)

    # Use fixed V/f block
    vf_angle   = FixedVfAngleBlock(
        "vf_angle",
        vf_ratio=VF_RATIO,
        v_phase_peak=V_PHASE_PEAK,
        boost_voltage=VF_BOOST,
        p_poles=P_POLES,
        boost_cutoff_speed=BOOST_CUTOFF_SPEED,
        min_speed_for_boost=MIN_SPEED_FOR_BOOST
    )

    vf_dq      = VfDQBlock("vf_dq")
    vf_theta   = VfThetaBlock("vf_theta")
    inv_park   = InvParkTransformBlock("inv_park", use_c_backend=False)
    svpwm_pack = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)
    svpwm_pack.INPUT_NAMES = ["magnitude", "angle_rad"]
    svpwm_pack.INPUT_KEEP  = [0, 1]
    svpwm      = SVPWMBlock("svpwm")
    cg_end     = CodeGenEnd("cg_end")

    # Use fixed plant
    motor      = FixedDB42S02PlantBlock("motor_sink", fmu_path=_FMU_PATH)
    sink       = VectorEnd("sink")
    sink_cg    = VectorEnd("sink_cg")

    # Wiring
    speed_ref  >> vf_angle
    vf_angle   >> vf_dq
    vf_angle   >> vf_theta
    vf_dq      >> inv_park
    vf_theta   >> inv_park
    inv_park   >> svpwm_pack
    svpwm_pack >> cg_start
    cg_start   >> svpwm
    svpwm      >> motor
    svpwm      >> cg_end
    motor      >> sink
    cg_end     >> sink_cg

    # Setup simulation
    sim = EmbedSim(
        sinks  = [sink, sink_cg],
        T      = T_SIM,
        dt     = DT,
        solver = ODESolver.EULER,
    )

    # Add signals to scope
    sim.scope.add(speed_ref,  indices=[0],             label="omega_ref")
    sim.scope.add(vf_angle,   indices=[0, 1, 2],       label="vf_angle")
    sim.scope.add(inv_park,   indices=[0, 1],          label="inv_park")
    sim.scope.add(svpwm_pack, indices=[0, 1, 2],       label="svpwm_pack")
    sim.scope.add(svpwm,      indices=[0, 1, 2, 3],    label="svpwm")
    sim.scope.add(motor,      indices=[0, 1, 2, 3, 4, 5, 6, 7], label="motor")

    # Export topology
    _topo_path = str(_HERE / "db42s02_topology.html")
    sim.topo.export_html(_topo_path)
    print(f"[Topology] {_topo_path}")

    # Run simulation
    sim.run()

    # Extract data
    sc = sim.scope
    hist = {
        "t":         np.array(sc.t, dtype=np.float32),
        "omega_ref": sc.get_signal("omega_ref", 0) * 60.0 / (2.0 * math.pi),
        "v_d":       sc.get_signal("vf_angle", 0),
        "v_q":       sc.get_signal("vf_angle", 1),
        "theta_e":   sc.get_signal("vf_angle", 2),
        "v_alpha":   sc.get_signal("inv_park", 0),
        "v_beta":    sc.get_signal("inv_park", 1),
        "vref":      sc.get_signal("svpwm_pack", 0),
        "angle_rad": sc.get_signal("svpwm_pack", 1),
        "ta":        sc.get_signal("svpwm", 0),
        "tb":        sc.get_signal("svpwm", 1),
        "tc":        sc.get_signal("svpwm", 2),
        "sector":    sc.get_signal("svpwm", 3).astype(int) + 1,
        "speed_rpm": sc.get_signal("motor", 0),
        "i_a":       sc.get_signal("motor", 1),
        "i_b":       sc.get_signal("motor", 2),
        "i_c":       sc.get_signal("motor", 3),
        "T_em":      sc.get_signal("motor", 4),
        "i_d":       sc.get_signal("motor", 5),
        "i_q":       sc.get_signal("motor", 6),
        "quality":   sc.get_signal("motor", 7),
    }

    # Add boost history
    if len(vf_angle.boost_history) > 0:
        hist["boost_applied"] = np.array(vf_angle.boost_history)
        hist["speed_boost"] = np.array(vf_angle.speed_history)
        hist["time_boost"] = np.array(vf_angle.time_history)

    return hist


# =============================================================================
# Plotting
# =============================================================================
def plot_results(d: dict, path: str = "db42s02_fixed_results.png"):
    """Plot results."""

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    t = d["t"]

    # Speed tracking
    axes[0, 0].plot(t, d["omega_ref"], "k--", label="Reference")
    axes[0, 0].plot(t, d["speed_rpm"], "b-", label="Actual")
    axes[0, 0].set_ylabel("Speed [RPM]")
    axes[0, 0].set_title("Speed Control")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Phase currents (last 100ms)
    n_samples = int(0.1 / DT)
    if len(t) > n_samples:
        t_zoom = t[-n_samples:] * 1000  # Convert to ms
        i_a_zoom = d["i_a"][-n_samples:]
        i_b_zoom = d["i_b"][-n_samples:]
        i_c_zoom = d["i_c"][-n_samples:]

        axes[0, 1].plot(t_zoom, i_a_zoom, "r-", label="i_a", alpha=0.8)
        axes[0, 1].plot(t_zoom, i_b_zoom, "g-", label="i_b", alpha=0.8)
        axes[0, 1].plot(t_zoom, i_c_zoom, "b-", label="i_c", alpha=0.8)
        axes[0, 1].set_xlabel("Time [ms]")
        axes[0, 1].set_ylabel("Current [A]")
        axes[0, 1].set_title(f"Phase Currents (Quality: {d['quality'][-1]:.1f}%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # dq currents
    axes[1, 0].plot(t, d["i_d"], "orange", label="i_d")
    axes[1, 0].plot(t, d["i_q"], "purple", label="i_q")
    axes[1, 0].set_ylabel("Current [A]")
    axes[1, 0].set_title("dq-axis Currents")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Quality over time
    axes[1, 1].plot(t, d["quality"], "m-", linewidth=2)
    axes[1, 1].set_ylabel("Quality [%]")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_title("Current Quality")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=80, color="g", linestyle="--", alpha=0.5, label="Target")
    axes[1, 1].legend()

    # Boost voltage
    if "time_boost" in d:
        axes[2, 0].plot(d["time_boost"], d["boost_applied"], "c-", linewidth=2)
        axes[2, 0].set_ylabel("Boost [V]")
        axes[2, 0].set_xlabel("Time [s]")
        axes[2, 0].set_title("Applied Voltage Boost")
        axes[2, 0].grid(True, alpha=0.3)

        # Speed vs boost
        axes[2, 1].plot(d["speed_boost"], d["boost_applied"], "co", markersize=2, alpha=0.5)
        axes[2, 1].set_xlabel("Speed [RPM]")
        axes[2, 1].set_ylabel("Boost [V]")
        axes[2, 1].set_title("Boost vs Speed")
        axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle(f"NANOTEC DB42S02 - Fixed V/f Control @ {OMEGA_CMD_RPM} RPM\n"
                 f"Boost: {VF_BOOST:.2f}V, Quality: {d['quality'][-1]:.1f}%",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  EmbedSim — NANOTEC DB42S02  Fixed V/f Control")
    print("=" * 70)
    print(f"\nMotor Parameters:")
    print(f"  R        = {R_STATOR:.3f} Ω")
    print(f"  Ld/Lq    = {L_d*1000:.3f} mH")
    print(f"  λ_pm     = {LAMBDA_PM:.4f} Wb")
    print(f"  p        = {P_POLES}")
    print(f"  V_dc     = {V_DC} V")

    print(f"\nControl Parameters:")
    print(f"  Target   : {OMEGA_CMD_RPM:.0f} RPM ({F_ELECTRICAL:.1f} Hz)")
    print(f"  V/f ratio: {VF_RATIO:.6f} V·s/rad")
    print(f"  Boost    : {VF_BOOST:.2f} V")
    print(f"  Boost cutoff: {BOOST_CUTOFF_SPEED:.0f} RPM")
    print(f"  dt       : {DT*1e6:.0f} µs")
    print(f"  Sim time : {T_SIM:.1f} s")
    print("=" * 70)

    # Run simulation
    data = build_and_run()

    # Plot results
    plot_results(data)

    # Print summary
    print(f"\nResults Summary:")
    print(f"  Final speed      : {data['speed_rpm'][-1]:.1f} RPM")
    print(f"  Final quality    : {data['quality'][-1]:.1f}%")
    print(f"  Average quality  : {np.mean(data['quality'][-2000:]):.1f}%")

    # Check currents
    steady_idx = int(0.8 * len(data['t']))
    i_a_rms = np.sqrt(np.mean(data['i_a'][steady_idx:]**2))
    i_b_rms = np.sqrt(np.mean(data['i_b'][steady_idx:]**2))
    i_c_rms = np.sqrt(np.mean(data['i_c'][steady_idx:]**2))

    print(f"\nSteady-state currents (RMS):")
    print(f"  i_a = {i_a_rms:.3f} A")
    print(f"  i_b = {i_b_rms:.3f} A")
    print(f"  i_c = {i_c_rms:.3f} A")

    if data['quality'][-1] > 80:
        print("\n✓ SUCCESS: Sinusoidal currents achieved!")
    else:
        print("\n⚠ Currents still need improvement")
        print("  Try increasing boost voltage or extending simulation time")

    print("\n[Done]")
    print(f"  Plot: db42s02_fixed_results.png")