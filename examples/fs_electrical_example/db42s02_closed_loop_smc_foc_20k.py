# db42s02_closed_loop_smc_foc_20k.py (corrected with proper INPUT_NAMES)

"""
db42s02_closed_loop_smc_foc_20k.py
==================================
EmbedSim  —  Closed-loop SMC FOC  —  NANOTEC DB42S02
=====================================================

Block diagram (Python-first, EmbedSim canonical order):
─────────────────────────────────────────────────────────────────
  CodeGenStart                    ← external inputs (feedback signals from sensors)

  SMCControllerBlock          [v_α, v_β]    ← Sliding Mode Control
       │
  SVPWMPackBlock              [Vref, α_rad] ← SVPWM packer
       │
  SVPWMBlock                  [ta, tb, tc, sector]
       │                     SVM_CalculateDutyCycle (C_CUSTOM_EMIT)
       │
  CodeGenEnd          ──→   EmbedSim_Output_T: {ta, tb, tc, sector}
       │
  DB42S02PlantBlock           PMSM_Motor.mo FMU
                              FMU inputs:
                                duty_a = ta
                                duty_b = tb
                                duty_c = tc
                                v_dc   = 17.0 V
                                T_load = schedule (0→5→20 mN·m)
─────────────────────────────────────────────────────────────────

CodeGen strategy (UPDATED)
────────────────
  CodeGenStart receives feedback signals from sensors (MCU):
    EmbedSim_Input_T = {
        omega_ref_mech,  // speed reference [rad/s]
        theta_m,         // mechanical angle [rad]
        ia, ib, ic       // phase currents [A]
    }

  Control algorithm runs entirely inside CodeGen region:
    - SMCControllerBlock: speed + current control
    - SVPWMPackBlock: generate voltage magnitude and angle
    - SVPWMBlock: calculate PWM duty cycles

  CodeGenEnd outputs PWM duties for inverter:
    EmbedSim_Output_T = {ta, tb, tc, sector}

  On the AURIX target the ISR:
    1. Reads sensors → fills EmbedSim_Input_T
    2. Calls EmbedSim_Step() → executes control algorithm
    3. Writes out.ta/tb/tc to GTM ATOM compare registers
"""

from __future__ import annotations

import sys, os, math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE = get_current_parent()
_ROOT = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"

for _p in (get_embedsim_import_path(), str(_FS_ELEC), str(_FS_ELEC / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep, VectorConstant
from embedsim.simulation_engine import VectorDelay
from embedsim.code_generator import CodeGenStart, CodeGenEnd

from motor_utility_blocks import SVPWMPackBlock
from svpwm_block import SVPWMBlock
from PMSM_MotorBlock import PMSM_MotorBlock
from smc_controller_block import SMCControllerBlock, _DB42S02
from pmsm_python_plant import PMSM_Python_Plant

_FMU_PATH = str(_FS_ELEC / "modelica" / "PMSM_Motor.fmu")

# =============================================================================
# Constants
# =============================================================================
V_DC = _DB42S02.SMC_V_DC
TARGET_RPM = 2000.0
TARGET_RADS_MECH = TARGET_RPM * 2.0 * math.pi / 60.0

T_SIM = 2.0
DT = 50e-6  # 20 kHz

T_LOAD_T1 = 0.5
T_LOAD_T2 = 1.2
T_LOAD_ZERO = 0.000
T_LOAD_LIGHT = 0.005
T_LOAD_HEAVY = 0.020

_MOTOR_OUT_SIZE = 8   # [rpm, ia, ib, ic, theta_m, T_em, id, iq]


# =============================================================================
# Plant Block
# =============================================================================
# =============================================================================
# Plant Block — pure Python, no FMU
# =============================================================================
class DB42S02PlantBlock(PMSM_Python_Plant):
    """
    DB42S02-specific plant.  Wraps PMSM_Python_Plant with:
      - DB42S02 motor parameters from _DB42S02
      - Load torque schedule applied inside compute_py
      - Output bus compatible with the rest of the simulation:
        [rpm, ia, ib, ic, theta_m, T_em, id, iq]   (8 elements)
    """
    TOPO_CATEGORY     = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm,ia,ib,ic,theta_m,Tem,id,iq]"

    _P = _DB42S02.SMC_P_POLES   # 4

    def __init__(self, name, **kwargs):
        super().__init__(
            name       = name,
            R          = _DB42S02.SMC_R_S,
            L_d        = _DB42S02.SMC_L_D,
            L_q        = _DB42S02.SMC_L_Q,
            lambda_pm  = _DB42S02.SMC_LAMBDA_PM,
            J          = _DB42S02.SMC_J_ROTOR,
            B_fric     = _DB42S02.SMC_B_FRICTION,
            p          = float(_DB42S02.SMC_P_POLES),
            v_dc       = V_DC,
        )

    def compute_py(self, t, dt, input_values=None):
        # Inject load torque schedule into input bus
        if t < T_LOAD_T1:
            t_load = T_LOAD_ZERO
        elif t < T_LOAD_T2:
            t_load = T_LOAD_LIGHT
        else:
            t_load = T_LOAD_HEAVY

        # Build input: [ta, tb, tc, v_dc, T_load]
        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                ta, tb, tc = float(v[0]), float(v[1]), float(v[2])

        augmented = [VectorSignal(
            np.array([ta, tb, tc, V_DC, t_load], dtype=DEFAULT_DTYPE))]
        return super().compute_py(t, dt, augmented)

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)



# =============================================================================
# CtrlPacker with INPUT_NAMES for code generation
# =============================================================================
class CtrlPacker(VectorBlock):
    """
    Packs feedback signals for simulation AND defines named input fields for CodeGen.

    The plant (DB42S02PlantBlock) now delivers theta_m already accumulated
    (unwrapped) at bus index [4].  CtrlPacker just routes it through and
    applies a speed-reference rate-limiter to prevent integrator wind-up.

    Bus from plant: [0]=rpm  [1]=ia  [2]=ib  [3]=ic  [4]=theta_m  [5]=T_em
    """
    INPUT_NAMES    = ["omega_ref_mech", "theta_m", "ia", "ib", "ic"]
    INPUT_KEEP     = [0, 1, 2, 3, 4]
    C_CODEGEN_EXCLUDE = True

    # Rate-limit: reach TARGET_RADS_MECH in 0.5 s
    _RAMP_TIME = 0.5
    _RAMP_RATE = TARGET_RADS_MECH / _RAMP_TIME   # rad/s²

    def __init__(self, name="ctrl_packer", **kw):
        super().__init__(name, **kw)
        self.output_label    = "[ω_ref_mech,θ_m,ia,ib,ic]"
        self._omega_ref_filt = 0.0    # rate-limited speed reference [rad/s]
        # theta_m unwrapper — accumulates continuous mechanical angle
        self._theta_m_prev:     float = 0.0
        self._theta_m_unwrapped: float = 0.0

    def reset(self):
        super().reset()
        self._omega_ref_filt     = 0.0
        self._theta_m_prev       = 0.0
        self._theta_m_unwrapped  = 0.0

    def compute_py(self, t, dt, input_values=None):
        # input_values[0] = motor bus  [rpm,ia,ib,ic,theta_m,T_em]
        # input_values[1] = speed_ref  [omega_ref_mech]
        m = (input_values[0].value if input_values and len(input_values) > 0
             else np.zeros(_MOTOR_OUT_SIZE, dtype=DEFAULT_DTYPE))
        r = (input_values[1].value if input_values and len(input_values) > 1
             else np.zeros(1, dtype=DEFAULT_DTYPE))

        # ── Rate-limit the speed reference ────────────────────────────────────
        omega_ref_target = float(r[0]) if len(r) > 0 else 0.0
        max_step = self._RAMP_RATE * dt
        self._omega_ref_filt += max(-max_step,
                                    min(max_step,
                                        omega_ref_target - self._omega_ref_filt))

        # ── Unwrap theta_m ───────────────────────────────────────────────────
        # PMSM_Python_Plant outputs theta_m = theta_e / p which wraps at
        # 2π/p ≈ 1.57 rad (p=4).  Without unwrapping the finite-difference
        # speed estimator in SMCControllerBlock sees large sign-flip spikes
        # every wrap and converges to omega_m_est ≈ 0.
        theta_m_raw = float(m[4]) if len(m) > 4 else 0.0
        delta = theta_m_raw - self._theta_m_prev
        # Bring delta into (-π, +π] to absorb the 2π/p wrap
        delta -= 2.0 * math.pi * math.floor((delta + math.pi) / (2.0 * math.pi))
        self._theta_m_unwrapped += delta
        self._theta_m_prev = theta_m_raw

        # theta_m at [4] is already accumulated/unwrapped by DB42S02PlantBlock
        self.output = VectorSignal(np.array([
            self._omega_ref_filt,                       # [0] ω_ref_mech (ramped)
            self._theta_m_unwrapped,                    # [1] θ_m (unwrapped, continuous)
            float(m[1]) if len(m) > 1 else 0.0,        # [2] ia
            float(m[2]) if len(m) > 2 else 0.0,        # [3] ib
            float(m[3]) if len(m) > 3 else 0.0,        # [4] ic
        ], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Build and Run
# =============================================================================
def build_and_run():
    print("=" * 70)
    print("  NANOTEC DB42S02 — Pure Sliding Mode FOC with Encoder | 20 kHz")
    print("=" * 70)
    print(f"  Target: {TARGET_RPM:.0f} RPM  ({TARGET_RADS_MECH:.1f} rad/s)")
    print(f"  Vdc={V_DC}V  dt={DT * 1e6:.0f}µs  T_sim={T_SIM}s")
    print()
    print("  SMC Controller Gains:")
    print(f"    Speed SMC: λ={_DB42S02.SMC_LAMBDA_W:.1f} rad/s, γ={_DB42S02.SMC_GAMMA_W:.1f} rad/s")
    print(f"               Ks=0.0120 N·m, φ=8.0 rad/s, η=0.05  (no-load tuned)")
    print(f"    Current SMC: Ks={_DB42S02.SMC_KS_I:.4f} V, φ={_DB42S02.SMC_PHI_I:.1f} A")
    print(f"    theta_m: direct from Python plant state (theta_e/p, no drift)")
    print()
    print("  Load Schedule:")
    print(f"    0 → {T_LOAD_LIGHT * 1e3:.0f} mN·m @ {T_LOAD_T1}s → {T_LOAD_HEAVY * 1e3:.0f} mN·m @ {T_LOAD_T2}s")
    print("=" * 70)

    # ── CodeGen blocks ──────────────────────────────────────────────────────
    # These blocks form the control algorithm that will run on the MCU
    cg_start = CodeGenStart("cg_start")  # Input boundary
    smc = SMCControllerBlock(
        "smc",
        SMC_V_DC   = V_DC,
        # Gains tuned for DB42S02 no-load / light-load bench condition.
        # Run smc_fmu_tuner.py --schedule to optimise for your load profile.
        SMC_KS_W   = 0.012,   # was 0.035 — reduced for low-inertia no-load
        SMC_ETA_W  = 0.05,    # was 0.10  — less damping avoids overshoot
        SMC_PHI_W  = 8.0,     # was 5.0   — wider BL reduces chattering
        SMC_KS_I   = 0.628,   # L_D * WC_I — unchanged
        SMC_PHI_I  = 0.5,     # unchanged
        dt_s       = DT,
        use_c_backend = False,   # Python backend: unwrapping speed estimator active
        integrator    = "tustin",
    )
    svpwm_pack = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)
    svpwm = SVPWMBlock("svpwm", use_c_backend = True)
    cg_end = CodeGenEnd("cg_end")  # Output boundary

    # ── Source blocks (simulation only) ─────────────────────────────────────
    speed_ref = VectorStep("speed_ref", step_time=0.0,
                           before_value=TARGET_RADS_MECH, after_value=TARGET_RADS_MECH)
    load_torque = VectorConstant("load_torque", value=T_LOAD_LIGHT)

    # ── Processing blocks (simulation only) ─────────────────────────────────
    motor = DB42S02PlantBlock("motor")
    motor_delay = VectorDelay("motor_delay", initial=[0.0] * _MOTOR_OUT_SIZE)
    ctrl = CtrlPacker("ctrl_packer")  # Packs feedback and defines named input fields

    # Sinks for simulation
    sink = VectorEnd("sink")
    sink_cg = VectorEnd("sink_cg")

    # ── CodeGen region wiring (complete control algorithm) ─────────────────
    # This entire chain runs on the target microcontroller
    # cg_start receives the named signals from ctrl_packer
    cg_start >> smc       # SMC controller (speed + current) - directly connected
    smc >> svpwm_pack     # Generate Vref and angle
    svpwm_pack >> svpwm   # SVPWM duty calculation
    svpwm >> cg_end       # Output PWM duties

    # ── Simulation wiring (outside CodeGen) ─────────────────────────────────
    # Feedback loop for simulation
    motor >> motor_delay
    motor_delay >> ctrl
    speed_ref >> ctrl
    ctrl >> cg_start  # Feedback enters CodeGen region

    # Plant connections
    cg_end >> motor  # PWM duties to plant
    load_torque >> motor  # Load torque

    # Sinks for data collection
    motor >> sink
    cg_end >> sink_cg

    # ── Simulation ──────────────────────────────────────────────────────────
    sim = EmbedSim(sinks=[sink, sink_cg], T=T_SIM, dt=DT, solver=ODESolver.EULER)

    print("\n[Topology]")
    sim.topo.print_console()

    # Export topology to HTML
    _wire_labels = {
        ("speed_ref", "ctrl_packer"): "ω_ref_mech [rad/s]",
        ("motor", "motor_delay"): "motor feedback",
        ("motor_delay", "ctrl_packer"): "delayed feedback",
        ("ctrl_packer", "cg_start"): "[ω_ref_mech, θ_m, ia, ib, ic] (named fields)",
        ("cg_start", "smc"): "Named input signals to SMC controller",
        ("smc", "svpwm_pack"): "[v_α, v_β]",
        ("svpwm_pack", "svpwm"): "[Vref, α_rad]",
        ("svpwm", "cg_end"): "[ta, tb, tc, sector]",
        ("cg_end", "motor"): "PWM duties to inverter",
        ("load_torque", "motor"): "T_load [N·m]",
        ("motor", "sink"): "plant outputs",
        ("cg_end", "sink_cg"): "PWM duties (logged)",
    }

    _topo_path = str(_HERE / "db42s02_smc_topology.html")
    sim.topo.export_html(_topo_path, wire_labels=_wire_labels)
    print(f"[Topology] {_topo_path}")

    # Add signals to scope
    sim.scope.add(speed_ref, indices=[0],             label="SpeedRef")
    sim.scope.add(smc,       indices=[0, 1],          label="Vab")
    sim.scope.add(svpwm_pack,indices=[0],             label="Vref")
    sim.scope.add(svpwm,     indices=[0, 1, 2, 3],    label="Duties")
    # motor output bus: [0]=rpm [1]=ia [2]=ib [3]=ic [4]=theta_m [5]=T_em [6]=id [7]=iq
    sim.scope.add(motor, indices=[0, 1, 2, 3, 5, 6, 7], label="Motor")

    print("\nRunning simulation...")
    sim.run()
    print(f"  Completed: {len(sim.scope.t)} steps")

    # ── StepGenerator ─────────────────────────────────────────────────────────
    print("\n[CodeGen] Calling cg_end.generate_step() …")
    result = cg_end.generate_step(
        cg_start=cg_start,
        output_dir=_ROOT,
        dt_hz=1.0 / DT,
        prefix="EmbedSim",
        write_files=True,
    )

    # Print generated files for verification
    if result:
        print(f"\n[CodeGen] Generated files in '{_ROOT / 'embedsim_gen'}':")
        print(f"  - embedsim_step.h")
        print(f"  - embedsim_step.c")
        print(f"  Input_T fields:")
        print(f"    - omega_ref_mech  (speed reference [rad/s])")
        print(f"    - theta_m         (mechanical angle [rad])")
        print(f"    - ia, ib, ic      (phase currents [A])")
        print(f"  Output_T fields: ta, tb, tc, sector")
        print(f"\n  The control algorithm (SMC + SVPWM) is now included in the generated code!")

    # Extract data for plotting
    sc = sim.scope
    t = np.array(sc.t, dtype=np.float32)

    # Motor signals — bus: [0]=rpm [1]=ia [2]=ib [3]=ic [4]=theta_m [5]=T_em
    # scope captured indices=[0,1,2,3,5] → positions 0..4
    #   position 0 → bus index 0 → speed_rpm
    #   position 4 → bus index 5 → T_em
    def _motor(pos):
        sig = sc.get_signal("Motor", pos)
        return sig if sig is not None else np.zeros(len(t), dtype=np.float32)

    motor_rpm = _motor(0)

    # Get SMC log data (speed values already stored in RPM by smc_controller_block.py)
    ld = smc.log_data

    # Interpolate SMC logs to simulation time grid
    def interp(key):
        if len(ld["t"]) > 1:
            return np.interp(t, ld["t"], ld[key]).astype(np.float32)
        return np.zeros(len(t), dtype=np.float32)

    def _scope(label, pos):
        sig = sc.get_signal(label, pos)
        return sig if sig is not None else np.zeros(len(t), dtype=np.float32)

    hist = {
        "t":            t,
        "speed_rpm":    motor_rpm,
        "omega_ref_rpm":interp("speed_ref"),
        "iq_ref":       interp("iq_ref"),
        "iq":           interp("iq"),
        "id":           interp("id"),
        "v_alpha":      _scope("Vab",    0),
        "v_beta":       _scope("Vab",    1),
        "vref":         _scope("Vref",   0),
        "ta":           _scope("Duties", 0),
        "tb":           _scope("Duties", 1),
        "tc":           _scope("Duties", 2),
        "sector":       _scope("Duties", 3),
        "torque":       _motor(4),   # scope pos 4 -> bus index 5 -> T_em
        "id_plant":     _motor(5),   # scope pos 5 -> bus index 6 -> id
        "iq_plant":     _motor(6),   # scope pos 6 -> bus index 7 -> iq
    }

    return hist


# =============================================================================
# Plot Results and Summary (unchanged)
# =============================================================================
def plot_results(d, path="db42s02_smc_foc_20k_results.png"):
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle(
        f"NANOTEC DB42S02 — Pure SMC FOC with Encoder | {TARGET_RPM:.0f} RPM | 20 kHz",
        fontsize=12, fontweight="bold")
    t = d["t"]

    # Speed tracking
    ax = axes[0, 0]
    ax.plot(t, d["omega_ref_rpm"], "k--", lw=1.5, label="ω_ref")
    ax.plot(t, d["speed_rpm"], "C0", lw=1.5, label="ω_actual")
    ax.axvline(T_LOAD_T1, color="orange", ls=":", lw=1.0)
    ax.axvline(T_LOAD_T2, color="red", ls=":", lw=1.0)
    ax.set_ylabel("Speed [RPM]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Speed tracking")
    ax.set_xlabel("t [s]")

    # Speed error
    ax = axes[0, 1]
    ax.plot(t, d["speed_rpm"] - d["omega_ref_rpm"], "C1", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(T_LOAD_T1, color="orange", ls=":", lw=1.0)
    ax.axvline(T_LOAD_T2, color="red", ls=":", lw=1.0)
    ax.set_ylabel("Error [RPM]")
    ax.grid(alpha=0.3)
    ax.set_title("Speed error")
    ax.set_xlabel("t [s]")

    # dq currents
    ax = axes[1, 0]
    ax.plot(t, d["iq_ref"], "k--", lw=1.2, label="iq_ref")
    ax.plot(t, d["iq"], "C0", lw=1.0, label="iq_meas")
    ax.plot(t, d["id"], "C5", lw=1.0, label="id_meas")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axhline(_DB42S02.SMC_I_MAX, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(-_DB42S02.SMC_I_MAX, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.set_ylabel("Current [A]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("dq currents — MTPA (id_ref=0)")
    ax.set_xlabel("t [s]")

    # id current
    ax = axes[1, 1]
    ax.plot(t, d["id"], "C5", lw=0.8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("id [A]")
    ax.grid(alpha=0.3)
    ax.set_title("id (should ≈ 0 — MTPA)")
    ax.set_xlabel("t [s]")

    # Voltage commands
    ax = axes[2, 0]
    ax.plot(t, d["v_alpha"], "C0", lw=0.8, label="v_α")
    ax.plot(t, d["v_beta"], "C1", lw=0.8, label="v_β")
    ax.set_ylabel("Voltage [V]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Stator voltage commands")
    ax.set_xlabel("t [s]")

    # Modulation index
    ax = axes[2, 1]
    ax.plot(t, d["vref"], "C5", lw=0.8)
    ax.axhline(0.95, color="red", ls="--", lw=0.8, alpha=0.7, label="clip=0.95")
    ax.set_ylabel("Vref [norm]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("SVPWM modulation index")
    ax.set_xlabel("t [s]")

    # PWM duties
    ax = axes[3, 0]
    ax.plot(t, d["ta"], "C3", lw=0.7, label="ta")
    ax.plot(t, d["tb"], "C2", lw=0.7, label="tb")
    ax.plot(t, d["tc"], "C1", lw=0.7, label="tc")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Duty")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("SVPWM duties")
    ax.set_xlabel("t [s]")

    # Torque
    ax = axes[3, 1]
    ax.plot(t, d["torque"] * 1000, "C4", lw=0.8, label="T_em")
    ax.axhline(T_LOAD_LIGHT * 1000, color="orange", ls=":", lw=1.0, alpha=0.7, label="light load")
    ax.axhline(T_LOAD_HEAVY * 1000, color="red", ls=":", lw=1.0, alpha=0.7, label="heavy load")
    ax.set_ylabel("Torque [mN·m]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Electromagnetic torque")
    ax.set_xlabel("t [s]")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


def print_summary(d):
    """Print performance summary."""
    n = len(d["t"])
    ss = int(0.8 * n)

    # Steady-state error
    err = float(np.mean(np.abs(d["speed_rpm"][ss:] - d["omega_ref_rpm"][ss:])))

    # Modulation index
    vrfx = float(np.max(d["vref"]))

    # Load step response
    after_load = d["t"] > T_LOAD_T2
    if np.any(after_load):
        avg_speed = float(np.mean(d["speed_rpm"][after_load][-100:]))
        speed_before = float(np.mean(d["speed_rpm"][d["t"] < T_LOAD_T2][-100:]))
        drop = speed_before - avg_speed if speed_before > avg_speed else 0.0
    else:
        avg_speed = float(d["speed_rpm"][-1])
        drop = 0.0

    # Current reference
    iqr_ss = float(np.mean(np.abs(d["iq_ref"][ss:])))

    print("\n" + "=" * 60)
    print("  Pure SMC FOC with Encoder — Performance Summary")
    print("=" * 60)
    print(f"  Final speed        : {d['speed_rpm'][-1]:.1f} RPM  (target {TARGET_RPM:.0f})")
    print(f"  SS speed error     : {err:.2f} RPM  (last 20% of run)")
    print(f"  Speed drop         : {drop:.0f} RPM at {T_LOAD_T2}s load step")
    print(f"  Recovery speed     : {avg_speed:.0f} RPM  (avg last 100 pts)")
    print(f"  iq_ref SS mean     : {iqr_ss:.3f} A  (expected {T_LOAD_HEAVY / _DB42S02.SMC_KT:.2f} A for 20 mN·m)")
    print(f"  Vref max           : {vrfx:.3f}  (clip 0.95)")
    print("=" * 60)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    data = build_and_run()
    plot_results(data)
    print_summary(data)
    print("\n[Done]  db42s02_smc_foc_20k_results.png")
    print("\n📁 Generated files:")
    print("   - embedsim_gen/embedsim_step.c")
    print("   - embedsim_gen/embedsim_step.h")
    print("   - db42s02_smc_topology.html")
    print("   - db42s02_smc_foc_20k_results.png")