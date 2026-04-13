"""
db42s02_openloop_fmu.py
=======================
EmbedSim  —  Open-loop V/f control  —  NANOTEC DB42S02
=======================================================

Library : fs_electrical_machines
Example : examples/pmsm_fmu_open_loop_example/

Block diagram (Python-first, EmbedSim canonical order)
───────────────────────────────────────────────────────
  CodeGenStart                       ← no upstream; Input_T is empty (_reserved)

  SpeedRampBlock      [omega_ref]    rad/s   (self-sourced)
       │
  VfAngleBlock        [v_d, v_q, theta_e]
       ├── VfDQBlock   [v_d, v_q]
       └── VfThetaBlock [theta_e]
                │
  InvParkTransformBlock  [v_alpha, v_beta]
       │
  SVPWMPackBlock      [Vref, angle_rad, Vdc]
       │  (indices 0,1 only cross CodeGen boundary)
  CodeGenStart ─► SVPWMBlock  [ta, tb, tc, sector]  ◄─ C_CUSTOM_EMIT
       │
  CodeGenEnd          → EmbedSim_Output_T { ta, tb, tc, sector }
       │
  DB42S02PlantBlock   PMSM_Plant_FMU.fmu
    inputs : duty_a=ta, duty_b=tb, duty_c=tc, v_dc=17 V, T_load (const)
    FMU outputs mapped to scope indices:
      [0] speed_rpm   [RPM]    [1] i_a  [A]   [2] i_b  [A]
      [3] i_c         [A]      [4] T_em [N·m]
───────────────────────────────────────────────────────

CodeGen boundary
────────────────
  cg_start : no upstream  → EmbedSim_Input_T  { _reserved }   (empty)
  cg_end   : svpwm output → EmbedSim_Output_T { ta, tb, tc, sector }

  SVPWMBlock carries the only C_CUSTOM_EMIT (SVM_CalculateDutyCycle —
  scalar ABI, normalised 1/Vdc duty output).
  No DutyPackBlock.  No hand-written C strings in this file.

  On the AURIX target the ISR writes:
      ATOM0_CH0_CM0 = (uint32_t)(out.ta * PWM_period)
      ATOM0_CH2_CM0 = (uint32_t)(out.tb * PWM_period)
      ATOM0_CH4_CM0 = (uint32_t)(out.tc * PWM_period)

Motor : NANOTEC DB42S02  (PMSM_Plant_FMU.fmu)
  R=0.19 Ω  Ld=Lq=0.125 mH  λ_pm=0.0014 Wb  J=2.4e-6 kg·m²  p=4  Vdc=17 V

Run:
    python db42s02_openloop_fmu.py
"""

from __future__ import annotations

import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Path bootstrap ──────────────────────────────────────────────────────────────
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

# ── EmbedSim ────────────────────────────────────────────────────────────────────
from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.code_generator import CodeGenStart, CodeGenEnd

# ── fs_electrical_machines ──────────────────────────────────────────────────────
from motor_utility_blocks import (
    SpeedRampBlock,
    VfAngleBlock,
    VfDQBlock,
    VfThetaBlock,
    SVPWMPackBlock,
)
from coordinate_transform_blocks import InvParkTransformBlock
from svpwm_block                   import SVPWMBlock
from PMSM_Plant_FMUBlock           import PMSM_Plant_FMUBlock  # FMU wrapper for PMSM_Plant_FMU.fmu

# ── FMU path ────────────────────────────────────────────────────────────────────
_FMU_PATH = str(_FS_ELEC / "modelica" / "PMSM_Plant_FMU.fmu")


# ═══════════════════════════════════════════════════════════════════════════════
# Motor / simulation constants  (NANOTEC DB42S02)
# ═══════════════════════════════════════════════════════════════════════════════
P_POLES        = 4
V_DC           = 17.0                               # V  — DC link voltage
V_PHASE_PEAK   = V_DC / math.sqrt(3.0)             # 9.815 V  — SVPWM maximum
OMEGA_M_RATED  = 8000.0 * 2.0 * math.pi / 60.0    # rad/s — rated mechanical speed
OMEGA_E_RATED  = P_POLES * OMEGA_M_RATED           # rad/s — rated electrical speed
VF_RATIO       = V_PHASE_PEAK / OMEGA_E_RATED      # V·s/rad  ≈ 2.929e-3

# Low-speed voltage boost — compensates stator resistive drop at low ω.
#   V0_BOOST = R × I_nom = 0.19 Ω × 1 A = 0.19 V
#   At rated speed this is only ~1.9 % of V_q — negligible.
#   Without boost at 400 RPM: V_q(no-boost) ≈ 0.49 V barely overcomes R×I.
R_STATOR       = 0.19                              # Ω
I_NOMINAL      = 1.0                               # A  — target no-load RMS current
VF_BOOST       = R_STATOR * I_NOMINAL              # V  = 0.19

# Shaft load — light bench-test condition.
#   T_LOAD = 0.01 N·m = 10 % of rated torque (0.1 N·m).
T_LOAD         = 0.01                              # N·m

# Simulation time window and step.
T_SIM          = 2.0                               # s   ramp (0.5 s) + steady (1.5 s)
DT             = 1e-4                              # s   100 µs step (10 kHz equivalent)

# Speed command.
#   400 RPM is below the R/L corner frequency (≈ 3630 RPM).
#   At 400 RPM: ω_e = 4·400·2π/60 = 167.6 rad/s
#               X_L = ω_e·L = 167.6·0.125e-3 ≈ 0.021 Ω  << R = 0.19 Ω
#   Resistive-dominant region → VF_BOOST critical.
OMEGA_CMD_RPM  = 400.0
OMEGA_CMD_RADS = OMEGA_CMD_RPM * 2.0 * math.pi / 60.0

# Ramp time.
#   accel = ω_cmd / t_ramp = 41.9 / 0.5 = 83.8 rad/s²
#   T_accel = J·α = 2.4e-6 · 83.8 ≈ 0.0002 N·m   << T_pullout ≈ 0.035 N·m
RAMP_TIME      = 0.5                               # s


# ═══════════════════════════════════════════════════════════════════════════════
# DB42S02PlantBlock  — simulation plant, NOT code-generated
# ═══════════════════════════════════════════════════════════════════════════════
class DB42S02PlantBlock(PMSM_Plant_FMUBlock):
    """
    NANOTEC DB42S02 plant adapter.

    Wraps PMSM_Plant_FMUBlock (FMU wrapper for PMSM_Plant_FMU.fmu) and
    translates the SVPWMBlock output bus [ta, tb, tc, sector] into
    the five FMU input scalars.

    Upstream  : SVPWMBlock → VectorSignal([ta, tb, tc, sector])

    FMU input bus (5 elements — PMSM_Plant_FMUBlock.INPUT_VARS):
        [0] duty_a = ta          normalised PWM duty [0, 1]
        [1] duty_b = tb
        [2] duty_c = tc
        [3] v_dc   = V_DC        17.0 V  — physical constant, never a CodeGen signal
        [4] T_load = T_LOAD      0.01 N·m — constant shaft load (10 % rated)

    FMU phase-voltage reconstruction (internal to PMSM_Plant_FMU.mo):
        v_x_leg   = duty_x · v_dc
        v_neutral = (v_a_leg + v_b_leg + v_c_leg) / 3
        v_x       = v_x_leg − v_neutral

    FMU OUTPUT_VARS (8 elements — PMSM_Plant_FMUBlock.OUTPUT_VARS):
        [0] rpm       [RPM]     [1] ia   [A]    [2] ib   [A]   [3] ic   [A]
        [4] theta_m   [rad]     [5] T_em [N·m]  [6] id_out [A] [7] iq_out [A]

    Scope output bus (5 elements) — indices for sim.scope.add():
        [0] speed_rpm   ← FMU 'rpm'   [RPM]
        [1] i_a         ← FMU 'ia'    [A]
        [2] i_b         ← FMU 'ib'    [A]
        [3] i_c         ← FMU 'ic'    [A]
        [4] T_em        ← FMU 'T_em'  [N·m]
    """

    # Exclude from CodeGen — this block is simulation-only
    TOPO_CATEGORY     = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm, ia, ib, ic, Tem]"

    def __init__(self, name: str) -> None:
        super().__init__(
            name     = name,
            fmu_path = _FMU_PATH,
        )
        # Convenience state mirrors (readable after each step)
        self.speed_rpm     = 0.0
        self.i_a = self.i_b = self.i_c = 0.0
        self.T_em          = 0.0
        self.theta_e_motor = 0.0
        print(f"[FMU] Loaded: {_FMU_PATH}")

    # Output index map — derived from PMSM_Plant_FMUBlock.OUTPUT_VARS:
    # ['rpm', 'ia', 'ib', 'ic', 'theta_m', 'T_em', 'id_out', 'iq_out']
    #   [0]    [1]   [2]   [3]   [4]        [5]     [6]       [7]
    _IDX_RPM     = 0   # rpm      [RPM]
    _IDX_IA      = 1   # ia       [A]
    _IDX_IB      = 2   # ib       [A]
    _IDX_IC      = 3   # ic       [A]
    _IDX_THETA_M = 4   # theta_m  [rad]  (mechanical angle = theta_e / p)
    _IDX_T_EM    = 5   # T_em     [N·m]

    def compute_py(self, t: float, dt: float, input_values=None):
        """
        Step the plant one time increment.

        Unpacks ta, tb, tc from the SVPWMBlock output bus (indices 0..2).
        Sector (index 3) is discarded — not an FMU input.
        v_dc and T_load are physical constants injected here, never part
        of the CodeGen region.
        """
        ta = tb = tc = 0.5                          # safe neutral default
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                ta, tb, tc = float(v[0]), float(v[1]), float(v[2])

        # Build the 5-element FMU input bus
        fmu_input = VectorSignal(
            np.array([ta, tb, tc, V_DC, T_LOAD], dtype=DEFAULT_DTYPE))
        super().compute_py(t, dt, [fmu_input])

        # Read FMU outputs into convenience mirrors
        # OUTPUT_VARS has 8 elements; _IDX_T_EM=5 is the last one we need
        if self.output is not None and len(self.output.value) > self._IDX_T_EM:
            ov = self.output.value
            self.speed_rpm     = float(ov[self._IDX_RPM])
            self.i_a           = float(ov[self._IDX_IA])
            self.i_b           = float(ov[self._IDX_IB])
            self.i_c           = float(ov[self._IDX_IC])
            self.theta_e_motor = float(ov[self._IDX_THETA_M])
            self.T_em          = float(ov[self._IDX_T_EM])
        else:
            self.speed_rpm = self.i_a = self.i_b = self.i_c = self.T_em = 0.0
            self.theta_e_motor = 0.0

        # Re-pack the scope output bus in a stable 5-element order
        self.output = VectorSignal(
            np.array([
                self.speed_rpm,   # [0]
                self.i_a,         # [1]
                self.i_b,         # [2]
                self.i_c,         # [3]
                self.T_em,        # [4]
            ], dtype=DEFAULT_DTYPE),
            self.name,
        )
        return self.output


# ═══════════════════════════════════════════════════════════════════════════════
# Build & run
# ═══════════════════════════════════════════════════════════════════════════════
def build_and_run() -> dict:
    """
    Wire all blocks, run the simulation, export topology HTML, call StepGenerator.

    CodeGen boundary
    ────────────────
    cg_start : no upstream      → EmbedSim_Input_T  { _reserved }   (empty)
    cg_end   : svpwm → cg_end  → EmbedSim_Output_T { ta, tb, tc, sector }

    Generated EmbedSim_step.c chain:
        SpeedRamp → VfAngle → VfDQ + VfTheta → InvPark
        → SVPWMPack → SVPWMBlock (SVM_CalculateDutyCycle)
        → out.ta / out.tb / out.tc / out.sector
    """

    # ── Instantiate ──────────────────────────────────────────────────────────
    cg_start   = CodeGenStart("cg_start")
    speed_ref  = SpeedRampBlock("speed_ref",
                                omega_target = OMEGA_CMD_RADS,
                                ramp_time    = RAMP_TIME)
    vf_angle   = VfAngleBlock("vf_angle",
                               vf_ratio     = VF_RATIO,
                               v_phase_peak = V_PHASE_PEAK,
                               v_boost      = VF_BOOST,
                               p_poles      = P_POLES)
    vf_dq      = VfDQBlock("vf_dq")
    vf_theta   = VfThetaBlock("vf_theta")
    inv_park   = InvParkTransformBlock("inv_park", use_c_backend=True)
    svpwm_pack = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)
    svpwm      = SVPWMBlock("svpwm")
    cg_end     = CodeGenEnd("cg_end")
    motor      = DB42S02PlantBlock("motor_sink")
    sink       = VectorEnd("sink")
    sink_cg    = VectorEnd("sink_cg")

    # SVPWMPackBlock exposes only magnitude (idx 0) and angle_rad (idx 1) across
    # the CodeGen boundary — Vdc at index 2 is an integration-layer constant.
    svpwm_pack.INPUT_NAMES = ["magnitude", "angle_rad"]
    svpwm_pack.INPUT_KEEP  = [0, 1]

    # ── Wire ─────────────────────────────────────────────────────────────────
    speed_ref  >> vf_angle
    vf_angle   >> vf_dq
    vf_angle   >> vf_theta
    vf_dq      >> inv_park          # port 0: [v_d, v_q]
    vf_theta   >> inv_park          # port 1: [theta_e]
    inv_park   >> svpwm_pack        # [v_alpha, v_beta] → [Vref, angle_rad, Vdc]
    svpwm_pack >> cg_start          # indices 0,1 become EmbedSim_Input_T
    cg_start   >> svpwm             # SVPWMBlock sits inside CodeGen region
    svpwm      >> cg_end            # [ta, tb, tc, sector] → CodeGen output
    svpwm      >> motor             # same bus feeds the FMU plant
    motor      >> sink
    cg_end     >> sink_cg

    # ── EmbedSim ─────────────────────────────────────────────────────────────
    sim = EmbedSim(
        sinks  = [sink, sink_cg],
        T      = T_SIM,
        dt     = DT,
        solver = ODESolver.EULER,
    )

    # ── Scope ─────────────────────────────────────────────────────────────────
    sim.scope.add(speed_ref,  indices=[0],             label="omega_ref")
    sim.scope.add(vf_angle,   indices=[0, 1, 2],       label="vf_angle")
    sim.scope.add(inv_park,   indices=[0, 1],           label="inv_park")
    sim.scope.add(svpwm_pack, indices=[0, 1, 2],       label="svpwm_pack")
    sim.scope.add(svpwm,      indices=[0, 1, 2, 3],    label="svpwm")
    sim.scope.add(motor,      indices=[0, 1, 2, 3, 4], label="motor")

    # ── Topology ──────────────────────────────────────────────────────────────
    _wire_labels = {
        ("speed_ref",  "vf_angle"):    "ω_ref [rad/s]",
        ("vf_angle",   "vf_dq"):       "[v_d, v_q, θ_e]",
        ("vf_angle",   "vf_theta"):    "[v_d, v_q, θ_e]",
        ("vf_dq",      "inv_park"):    "[v_d, v_q]",
        ("vf_theta",   "inv_park"):    "θ_e",
        ("inv_park",   "svpwm_pack"):  "[v_α, v_β]",
        ("svpwm_pack", "cg_start"):    "[Vref, α_rad]",
        ("cg_start",   "svpwm"):       "[Vref, α_rad]",
        ("svpwm",      "cg_end"):      "[ta, tb, tc, sector]",
        ("svpwm",      "motor_sink"):  "[ta, tb, tc, sector]",
        ("cg_end",     "sink_cg"):     "[ta, tb, tc, sector]",
        ("motor_sink", "sink"):        "[rpm, ia, ib, ic, Tem]",
    }

    print("\n[Topology] Signal-flow diagram:")
    sim.topo.print_console()
    _topo_path = str(_HERE / "db42s02_topology.html")
    sim.topo.export_html(_topo_path, wire_labels=_wire_labels)
    print(f"[Topology] {_topo_path}")

    # ── Run ───────────────────────────────────────────────────────────────────
    sim.run()

    # ── Extract signals ───────────────────────────────────────────────────────
    sc = sim.scope
    hist = {
        "t":         np.array(sc.t,  dtype=np.float32),
        # Speed reference: convert rad/s → RPM for plotting
        "omega_ref": sc.get_signal("omega_ref",  0) * 60.0 / (2.0 * math.pi),
        # VfAngle outputs
        "v_d":       sc.get_signal("vf_angle",   0),
        "v_q":       sc.get_signal("vf_angle",   1),
        "theta_e":   sc.get_signal("vf_angle",   2),
        # InvPark outputs
        "v_alpha":   sc.get_signal("inv_park",   0),
        "v_beta":    sc.get_signal("inv_park",   1),
        # SVPWMPack outputs
        "vref":      sc.get_signal("svpwm_pack", 0),
        "angle_rad": sc.get_signal("svpwm_pack", 1),
        # SVPWMBlock outputs — sector is 0-indexed internally; add 1 for display
        "ta":        sc.get_signal("svpwm",      0),
        "tb":        sc.get_signal("svpwm",      1),
        "tc":        sc.get_signal("svpwm",      2),
        "sector":    sc.get_signal("svpwm",      3).astype(int) + 1,
        # FMU plant outputs (scope bus index order defined in DB42S02PlantBlock)
        "speed_rpm": sc.get_signal("motor",      0),
        "i_a":       sc.get_signal("motor",      1),
        "i_b":       sc.get_signal("motor",      2),
        "i_c":       sc.get_signal("motor",      3),
        "T_em":      sc.get_signal("motor",      4),
    }

    # ── StepGenerator ─────────────────────────────────────────────────────────
    print("\n[CodeGen] Calling cg_end.generate_step() …")
    cg_end.generate_step(
        cg_start    = cg_start,
        output_dir  = _ROOT,
        dt_hz       = 1.0 / DT,
        prefix      = "EmbedSim",
        write_files = True,
    )

    return hist


# ═══════════════════════════════════════════════════════════════════════════════
# Static plots
# ═══════════════════════════════════════════════════════════════════════════════
_SECTOR_COLORS = [
    "#FF595E", "#FF924C", "#FFCA3A",
    "#8AC926", "#1982C4", "#6A4C93",
]


def plot_results(d: dict, path: str = "db42s02_openloop_results.png") -> None:
    """Four-panel static results: speed, currents, duty cycles, sector+v_q."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        f"NANOTEC DB42S02 — Open-loop V/f  "
        f"(cmd {OMEGA_CMD_RPM:.0f} RPM | V_dc {V_DC} V | p={P_POLES})",
        fontsize=13, fontweight="bold",
    )
    t = d["t"]

    axes[0].plot(t, d["omega_ref"], "k--", lw=1.2, label="ω_ref [RPM]")
    axes[0].plot(t, d["speed_rpm"], "C0",  lw=1.5, label="ω_motor FMU [RPM]")
    axes[0].set_ylabel("Speed [RPM]")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Motor speed  (PMSM_Plant_FMU.fmu plant)")

    axes[1].plot(t, d["i_a"], "C3", lw=0.9, label="i_a")
    axes[1].plot(t, d["i_b"], "C2", lw=0.9, label="i_b")
    axes[1].plot(t, d["i_c"], "C1", lw=0.9, label="i_c")
    axes[1].set_ylabel("Current [A]")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Phase currents — FMU sensor output")

    axes[2].plot(t, d["ta"], "C3", lw=0.8, label="ta → GTM ATOM0 CH0")
    axes[2].plot(t, d["tb"], "C2", lw=0.8, label="tb → GTM ATOM0 CH2")
    axes[2].plot(t, d["tc"], "C1", lw=0.8, label="tc → GTM ATOM0 CH4")
    axes[2].set_ylabel("Duty [0–1]")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("SVM duty cycles  (SVPWMBlock → AURIX GTM ATOM)")

    ax4 = axes[3]
    ax4b = ax4.twinx()
    ax4.step(t, d["sector"], where="post", color="C5", lw=1.2, label="SVPWM sector")
    ax4b.plot(t, d["v_q"], "C0--", lw=1.0, label="v_q [V]")
    ax4.set_ylabel("Sector [1–6]", color="C5")
    ax4b.set_ylabel("v_q [V]", color="C0")
    ax4.set_ylim(0, 7)
    ax4.set_yticks([1, 2, 3, 4, 5, 6])
    ax4.set_xlabel("Time [s]")
    ax4.grid(alpha=0.3)
    ax4.set_title("SVPWM sector + v_q")
    l1, n1 = ax4.get_legend_handles_labels()
    l2, n2 = ax4b.get_legend_handles_labels()
    ax4.legend(l1 + l2, n1 + n2, fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


def plot_phasor_static(d: dict, path: str = "db42s02_phasor_sectors.png") -> None:
    """Static α-β SVPWM hexagon with full trajectory and final phasor arrow."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_title(
        f"SVPWM hexagon — α-β plane\n"
        f"NANOTEC DB42S02  {OMEGA_CMD_RPM:.0f} RPM  open-loop V/f",
        fontsize=12, fontweight="bold",
    )
    R = 2.0 / 3.0
    for s in range(6):
        a0   = math.radians(60 * s)
        a1   = math.radians(60 * (s + 1))
        angs = np.linspace(a0, a1, 30)
        ax.fill(
            np.concatenate([[0], R * np.cos(angs), [0]]),
            np.concatenate([[0], R * np.sin(angs), [0]]),
            color=_SECTOR_COLORS[s], alpha=0.22,
        )
        am = (a0 + a1) / 2.0
        ax.text(
            0.45 * math.cos(am), 0.45 * math.sin(am),
            f"S{s+1}", ha="center", va="center",
            fontsize=13, fontweight="bold", color=_SECTOR_COLORS[s],
        )
    ax.plot(
        [R * math.cos(k * math.pi / 3) for k in range(7)],
        [R * math.sin(k * math.pi / 3) for k in range(7)],
        "k-", lw=1.5,
    )
    for k in range(6):
        ax.text(
            R * math.cos(k * math.pi / 3) * 1.08,
            R * math.sin(k * math.pi / 3) * 1.08,
            f"V{k+1}", ha="center", fontsize=9, fontweight="bold",
        )
    sc_norm = 1.0 / V_DC
    ax.plot(
        d["v_alpha"] * sc_norm, d["v_beta"] * sc_norm,
        "navy", lw=0.5, alpha=0.35, label="α-β trajectory",
    )
    vx = float(d["v_alpha"][-1]) * sc_norm
    vy = float(d["v_beta"][-1])  * sc_norm
    ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=2.5))
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("α  (normalised to V_dc)")
    ax.set_ylabel("β")
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.85, 0.85)
    ax.grid(alpha=0.25)
    ax.legend(
        handles=[
            mpatches.Patch(color=_SECTOR_COLORS[i], alpha=0.5, label=f"Sector {i+1}")
            for i in range(6)
        ],
        fontsize=8, loc="lower right",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Animation helpers
# ═══════════════════════════════════════════════════════════════════════════════

# ── Dark-theme palette ─────────────────────────────────────────────────────────
_BG_FIG  = "#0e1117"
_BG_AX   = "#1a1d27"
_GRID_C  = "#2e3145"
_TXT     = "#e0e4f0"
_CURSOR  = "#FFD700"
_ON_CLR  = "#00E676"    # IGBT ON  — vivid green
_OFF_CLR = "#252836"    # IGBT OFF — dark slate
_ON_TXT  = "#00E676"
_OFF_TXT = "#4a4e6a"
_WIRE    = "#5a6080"
_DC_POS  = "#FF5252"    # +Vdc rail
_DC_NEG  = "#448AFF"    # GND  rail
_PH_CLR  = {"a": "#FF595E", "b": "#8AC926", "c": "#1982C4"}


def _style_ax(ax) -> None:
    ax.set_facecolor(_BG_AX)
    ax.tick_params(colors=_TXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID_C)
    ax.xaxis.label.set_color(_TXT)
    ax.yaxis.label.set_color(_TXT)
    ax.title.set_color(_TXT)
    ax.grid(color=_GRID_C, linewidth=0.5, alpha=0.7)


def _make_sector_patch(s: int, R: float = 2.0 / 3.0) -> mpatches.Polygon:
    """Filled wedge for SVPWM sector *s* (0-indexed), initially invisible."""
    a0   = math.radians(60 * s)
    a1   = math.radians(60 * (s + 1))
    angs = np.linspace(a0, a1, 40)
    xs   = np.concatenate([[0.0], R * np.cos(angs), [0.0]])
    ys   = np.concatenate([[0.0], R * np.sin(angs), [0.0]])
    return mpatches.Polygon(
        np.column_stack([xs, ys]),
        closed=True, facecolor=_SECTOR_COLORS[s],
        alpha=0.0, edgecolor="none", zorder=1,
    )


def _draw_inverter_bridge(ax) -> dict:
    """
    Draw the static 3-phase 2-level VSI schematic on *ax*.

    Data coordinates: 0..1 × 0..1 (axis("off")).

    Returns mutable artist references:
        "Q{a/b/c}_{H/L}"       FancyBboxPatch — IGBT body
        "T{a/b/c}_{H/L}"       Text           — IGBT label
        "duty_label_{a/b/c}"   Text           — duty readout
        "I_label_{a/b/c}"      Text           — current readout
        "sector_badge"          Text           — active sector
        "vdc_label"             Text           — Vdc value

    Switch logic (symmetrical triangle carrier):
        duty_x > 0.5  →  Q_high ON,  Q_low OFF
        duty_x ≤ 0.5  →  Q_high OFF, Q_low ON

    IEC / AURIX GTM app-note numbering:
        Q1(Qa_H) Q3(Qb_H) Q5(Qc_H)   ← upper rail
        Q4(Qa_L) Q6(Qb_L) Q2(Qc_L)   ← lower rail
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("3-Phase 2-Level VSI  (AURIX GTM ATOM0)",
                 fontsize=9, fontweight="bold", color=_TXT, pad=5)

    arts: dict = {}

    Y_POS = 0.90
    Y_NEG = 0.08
    Y_MID = (Y_POS + Y_NEG) / 2      # 0.49

    # DC rails
    ax.plot([0.05, 0.95], [Y_POS, Y_POS], color=_DC_POS, lw=2.8, zorder=2)
    ax.plot([0.05, 0.95], [Y_NEG, Y_NEG], color=_DC_NEG, lw=2.8, zorder=2)
    ax.text(0.04, Y_POS, "+Vdc", color=_DC_POS, fontsize=7,
            va="center", ha="right", fontweight="bold")
    ax.text(0.04, Y_NEG, "GND",  color=_DC_NEG, fontsize=7,
            va="center", ha="right", fontweight="bold")
    arts["vdc_label"] = ax.text(
        0.50, 0.975, f"V_dc = {V_DC:.1f} V",
        color=_TXT, fontsize=8.5, ha="center", va="top", fontweight="bold",
    )

    PH_X    = {"a": 0.22, "b": 0.50, "c": 0.78}
    SW_W    = 0.14
    SW_H    = 0.115
    Y_H_CTR = (Y_POS + Y_MID) / 2 + 0.01
    Y_L_CTR = (Y_MID + Y_NEG) / 2 - 0.01
    IGBT_H  = {"a": "Q1 (Qa↑)", "b": "Q3 (Qb↑)", "c": "Q5 (Qc↑)"}
    IGBT_L  = {"a": "Q4 (Qa↓)", "b": "Q6 (Qb↓)", "c": "Q2 (Qc↓)"}
    CH_LBL  = {"a": "CH0",      "b": "CH2",       "c": "CH4"}

    for ph in ("a", "b", "c"):
        cx = PH_X[ph]

        # Upper IGBT
        box_h = FancyBboxPatch(
            (cx - SW_W / 2, Y_H_CTR - SW_H / 2), SW_W, SW_H,
            boxstyle="round,pad=0.006",
            facecolor=_OFF_CLR, edgecolor=_WIRE, lw=1.3, zorder=4,
        )
        ax.add_patch(box_h)
        arts[f"Q{ph}_H"] = box_h
        arts[f"T{ph}_H"] = ax.text(
            cx, Y_H_CTR, IGBT_H[ph],
            color=_OFF_TXT, fontsize=6.5, ha="center", va="center",
            fontweight="bold", zorder=5,
        )

        # Lower IGBT
        box_l = FancyBboxPatch(
            (cx - SW_W / 2, Y_L_CTR - SW_H / 2), SW_W, SW_H,
            boxstyle="round,pad=0.006",
            facecolor=_OFF_CLR, edgecolor=_WIRE, lw=1.3, zorder=4,
        )
        ax.add_patch(box_l)
        arts[f"Q{ph}_L"] = box_l
        arts[f"T{ph}_L"] = ax.text(
            cx, Y_L_CTR, IGBT_L[ph],
            color=_OFF_TXT, fontsize=6.5, ha="center", va="center",
            fontweight="bold", zorder=5,
        )

        # Vertical wires
        ax.plot([cx, cx], [Y_POS,              Y_H_CTR + SW_H / 2], color=_WIRE, lw=1.3, zorder=3)
        ax.plot([cx, cx], [Y_H_CTR - SW_H / 2, Y_MID             ], color=_WIRE, lw=1.3, zorder=3)
        ax.plot([cx, cx], [Y_MID,              Y_L_CTR + SW_H / 2], color=_WIRE, lw=1.3, zorder=3)
        ax.plot([cx, cx], [Y_L_CTR - SW_H / 2, Y_NEG             ], color=_WIRE, lw=1.3, zorder=3)

        # Phase midpoint node
        dot = mpatches.Circle(
            (cx, Y_MID), radius=0.016,
            facecolor=_PH_CLR[ph], edgecolor="white", lw=0.8, zorder=6,
        )
        ax.add_patch(dot)
        ax.text(cx, Y_MID + 0.07, f"Ph-{ph.upper()}\n{CH_LBL[ph]}",
                color=_PH_CLR[ph], fontsize=6, ha="center", va="bottom",
                fontweight="bold")

        # Duty readout above upper switch
        arts[f"duty_label_{ph}"] = ax.text(
            cx, Y_POS + 0.045, f"t{ph}=0.500",
            color=_PH_CLR[ph], fontsize=6.5, ha="center", va="bottom",
        )
        # Current readout below lower switch
        arts[f"I_label_{ph}"] = ax.text(
            cx, Y_NEG - 0.07, f"i{ph}=0.00A",
            color=_PH_CLR[ph], fontsize=6.5, ha="center", va="top",
        )

    # Motor neutral stub → star point
    for ph in ("a", "b", "c"):
        cx = PH_X[ph]
        ax.annotate(
            "", xy=(0.96, Y_MID), xytext=(cx + SW_W / 2 + 0.005, Y_MID),
            arrowprops=dict(arrowstyle="-", color=_PH_CLR[ph], lw=1.1),
            zorder=3,
        )
    star = mpatches.Circle(
        (0.96, Y_MID), radius=0.013,
        facecolor="#777", edgecolor="white", lw=0.6, zorder=6,
    )
    ax.add_patch(star)
    ax.text(0.975, Y_MID, "N", color="#aaa", fontsize=6, va="center", ha="left")

    arts["sector_badge"] = ax.text(
        0.50, Y_MID + 0.005, "Sector –",
        color=_TXT, fontsize=9.5, ha="center", va="center",
        fontweight="bold", zorder=10,
    )
    return arts


def animate_phasor(
    d: dict,
    path: str        = "db42s02_phasor_anim.gif",
    n_frames: int    = 200,
    interval_ms: int = 50,
    save_mp4: bool   = False,
) -> None:
    """
    Six-panel animation (3 rows × 2 cols):

        [0,0]  α-β SVPWM phasor hexagon    [0,1]  3-phase VSI switch states
        [1,0]  PWM duty cycles ta/tb/tc    [1,1]  Phase currents ia/ib/ic
        [2,0]  Motor speed [RPM]            [2,1]  Electromagnetic torque T_em

    Parameters
    ----------
    d            dict returned by build_and_run()
    path         output filename  (.gif or .mp4)
    n_frames     number of animation frames
    interval_ms  milliseconds between frames
    save_mp4     True → FFMpegWriter  |  False → PillowWriter (.gif)
    """
    t         = d["t"]
    N         = len(t)
    sc        = 1.0 / V_DC
    frame_idx = np.linspace(0, N - 1, n_frames, dtype=int)
    va_n      = d["v_alpha"] * sc
    vb_n      = d["v_beta"]  * sc
    trail_len = max(1, N // 30)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 9), facecolor=_BG_FIG)
    fig.patch.set_facecolor(_BG_FIG)
    gs = fig.add_gridspec(
        3, 2, hspace=0.46, wspace=0.28,
        left=0.06, right=0.97, top=0.93, bottom=0.06,
    )
    ax_hex  = fig.add_subplot(gs[0, 0])
    ax_inv  = fig.add_subplot(gs[0, 1])
    ax_duty = fig.add_subplot(gs[1, 0])
    ax_curr = fig.add_subplot(gs[1, 1])
    ax_spd  = fig.add_subplot(gs[2, 0])
    ax_tem  = fig.add_subplot(gs[2, 1])

    for ax in (ax_hex, ax_duty, ax_curr, ax_spd, ax_tem):
        _style_ax(ax)

    fig.suptitle(
        f"NANOTEC DB42S02 — Open-loop V/f  |  "
        f"cmd {OMEGA_CMD_RPM:.0f} RPM  |  V_dc {V_DC} V  |  p = {P_POLES}",
        color=_TXT, fontsize=11, fontweight="bold",
    )

    # ── ax_hex: α-β phasor hexagon ────────────────────────────────────────────
    R = 2.0 / 3.0
    ax_hex.set_aspect("equal")
    ax_hex.set_xlim(-0.88, 0.88); ax_hex.set_ylim(-0.88, 0.88)
    ax_hex.set_xlabel("α  (norm. to V_dc)", fontsize=8)
    ax_hex.set_ylabel("β", fontsize=8)
    ax_hex.set_title("SVPWM  α-β phasor", fontsize=9, fontweight="bold")

    sec_patches = [_make_sector_patch(s, R) for s in range(6)]
    for sp in sec_patches:
        ax_hex.add_patch(sp)

    hx = [R * math.cos(k * math.pi / 3) for k in range(7)]
    hy = [R * math.sin(k * math.pi / 3) for k in range(7)]
    ax_hex.plot(hx, hy, color="#888", lw=1.2, zorder=2)
    for k in range(6):
        ax_hex.text(
            R * math.cos(k * math.pi / 3) * 1.14,
            R * math.sin(k * math.pi / 3) * 1.14,
            f"V{k+1}", ha="center", va="center",
            fontsize=7, fontweight="bold", color="#aaa", zorder=3,
        )
    for s in range(6):
        am = math.radians(60 * s + 30)
        ax_hex.text(
            0.44 * math.cos(am), 0.44 * math.sin(am), f"S{s+1}",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=_SECTOR_COLORS[s], alpha=0.55, zorder=4,
        )
    ax_hex.axhline(0, color=_GRID_C, lw=0.5)
    ax_hex.axvline(0, color=_GRID_C, lw=0.5)
    ax_hex.plot(va_n, vb_n, color="navy", lw=0.4, alpha=0.18, zorder=5)

    phasor = FancyArrowPatch(
        posA=(0, 0), posB=(float(va_n[0]), float(vb_n[0])),
        arrowstyle="-|>", mutation_scale=15,
        color="#FF4040", lw=2.4, zorder=10,
    )
    ax_hex.add_patch(phasor)
    (trail,)       = ax_hex.plot([], [], color="#FF7070", lw=1.5, alpha=0.8, zorder=9)
    hex_sec_txt    = ax_hex.text(-0.83,  0.80, "Sector –",    color=_TXT,   fontsize=9, fontweight="bold", zorder=11)
    hex_time_txt   = ax_hex.text(-0.83, -0.80, "t = 0.000 s", color="#888", fontsize=8, zorder=11)

    # ── ax_inv: 3-phase inverter bridge ──────────────────────────────────────
    inv = _draw_inverter_bridge(ax_inv)

    # ── ax_duty: PWM duty cycles ──────────────────────────────────────────────
    for key, lbl, col in (
        ("ta", "ta → CH0", _PH_CLR["a"]),
        ("tb", "tb → CH2", _PH_CLR["b"]),
        ("tc", "tc → CH4", _PH_CLR["c"]),
    ):
        ax_duty.plot(t, d[key], color=col, lw=0.8, alpha=0.50, label=lbl)
    ax_duty.axhline(0.5, color="#555", lw=0.8, ls="--", label="50 % threshold")
    ax_duty.set_ylim(-0.05, 1.05)
    ax_duty.set_xlabel("Time [s]", fontsize=8)
    ax_duty.set_ylabel("Duty [0–1]", fontsize=8)
    ax_duty.set_title("PWM duty cycles", fontsize=9, fontweight="bold")
    ax_duty.legend(fontsize=7, loc="upper left",
                   facecolor=_BG_AX, edgecolor=_GRID_C, labelcolor=_TXT)
    duty_cur  = ax_duty.axvline(t[0], color=_CURSOR, lw=1.2, alpha=0.85)
    duty_dots = [
        ax_duty.plot([], [], "o", color=c, ms=5, zorder=10)[0]
        for c in (_PH_CLR["a"], _PH_CLR["b"], _PH_CLR["c"])
    ]

    # ── ax_curr: phase currents ───────────────────────────────────────────────
    for key, lbl, col in (
        ("i_a", "ia", _PH_CLR["a"]),
        ("i_b", "ib", _PH_CLR["b"]),
        ("i_c", "ic", _PH_CLR["c"]),
    ):
        ax_curr.plot(t, d[key], color=col, lw=0.8, alpha=0.50, label=lbl)
    ax_curr.set_xlabel("Time [s]", fontsize=8)
    ax_curr.set_ylabel("Current [A]", fontsize=8)
    ax_curr.set_title("Phase currents — FMU sensor", fontsize=9, fontweight="bold")
    ax_curr.legend(fontsize=7, loc="upper left",
                   facecolor=_BG_AX, edgecolor=_GRID_C, labelcolor=_TXT)
    curr_cur  = ax_curr.axvline(t[0], color=_CURSOR, lw=1.2, alpha=0.85)
    curr_dots = [
        ax_curr.plot([], [], "o", color=c, ms=5, zorder=10)[0]
        for c in (_PH_CLR["a"], _PH_CLR["b"], _PH_CLR["c"])
    ]

    # ── ax_spd: motor speed ───────────────────────────────────────────────────
    ax_spd.plot(t, d["omega_ref"],  color="#888",    lw=1.0, ls="--", label="ω_ref")
    ax_spd.plot(t, d["speed_rpm"],  color="#FFCA3A", lw=1.2, label="ω_motor")
    ax_spd.set_xlabel("Time [s]", fontsize=8)
    ax_spd.set_ylabel("Speed [RPM]", fontsize=8)
    ax_spd.set_title("Motor speed (FMU plant)", fontsize=9, fontweight="bold")
    ax_spd.legend(fontsize=7, loc="upper left",
                  facecolor=_BG_AX, edgecolor=_GRID_C, labelcolor=_TXT)
    spd_cur  = ax_spd.axvline(t[0], color=_CURSOR, lw=1.2, alpha=0.85)
    (spd_dot,) = ax_spd.plot([], [], "o", color="#FFCA3A", ms=5, zorder=10)

    # ── ax_tem: electromagnetic torque ────────────────────────────────────────
    ax_tem.plot(t, d["T_em"], color="#FF924C", lw=1.0, alpha=0.65, label="T_em")
    ax_tem.set_xlabel("Time [s]", fontsize=8)
    ax_tem.set_ylabel("Torque [N·m]", fontsize=8)
    ax_tem.set_title("Electromagnetic torque", fontsize=9, fontweight="bold")
    ax_tem.legend(fontsize=7, loc="upper left",
                  facecolor=_BG_AX, edgecolor=_GRID_C, labelcolor=_TXT)
    tem_cur  = ax_tem.axvline(t[0], color=_CURSOR, lw=1.2, alpha=0.85)
    (tem_dot,) = ax_tem.plot([], [], "o", color="#FF924C", ms=5, zorder=10)

    # ── Update callback ───────────────────────────────────────────────────────
    _prev_sec = [-1]

    def _update(frame: int):
        i   = frame_idx[frame]
        ti  = float(t[i])
        vax = float(va_n[i])
        vbx = float(vb_n[i])
        sec = int(d["sector"][i]) - 1      # back to 0-indexed for patch array
        ta_ = float(d["ta"][i])
        tb_ = float(d["tb"][i])
        tc_ = float(d["tc"][i])
        ia_ = float(d["i_a"][i])
        ib_ = float(d["i_b"][i])
        ic_ = float(d["i_c"][i])

        # Phasor + trail
        phasor.set_positions((0, 0), (vax, vbx))
        i0 = max(0, i - trail_len)
        trail.set_data(va_n[i0:i + 1], vb_n[i0:i + 1])
        hex_time_txt.set_text(f"t = {ti:.3f} s")

        # Sector highlight
        if sec != _prev_sec[0]:
            col = _SECTOR_COLORS[sec]
            for s, sp in enumerate(sec_patches):
                sp.set_alpha(0.35 if s == sec else 0.0)
            lbl = f"Sector {sec + 1}"
            hex_sec_txt.set_text(lbl); hex_sec_txt.set_color(col)
            inv["sector_badge"].set_text(lbl); inv["sector_badge"].set_color(col)
            _prev_sec[0] = sec

        # Inverter switch states
        duties   = {"a": ta_, "b": tb_, "c": tc_}
        currents = {"a": ia_, "b": ib_, "c": ic_}
        for ph, tx in duties.items():
            on_h = tx > 0.5
            inv[f"Q{ph}_H"].set_facecolor(_ON_CLR  if on_h     else _OFF_CLR)
            inv[f"T{ph}_H"].set_color    (_ON_TXT  if on_h     else _OFF_TXT)
            inv[f"Q{ph}_L"].set_facecolor(_ON_CLR  if not on_h else _OFF_CLR)
            inv[f"T{ph}_L"].set_color    (_ON_TXT  if not on_h else _OFF_TXT)
            inv[f"duty_label_{ph}"].set_text(f"t{ph}={tx:.3f}")
            inv[f"I_label_{ph}"].set_text(f"i{ph}={currents[ph]:+.2f}A")

        # Time cursors
        for cur in (duty_cur, curr_cur, spd_cur, tem_cur):
            cur.set_xdata([ti, ti])

        # Moving dots
        for dot, key in zip(duty_dots, ("ta", "tb", "tc")):
            dot.set_data([ti], [float(d[key][i])])
        for dot, key in zip(curr_dots, ("i_a", "i_b", "i_c")):
            dot.set_data([ti], [float(d[key][i])])
        spd_dot.set_data([ti], [float(d["speed_rpm"][i])])
        tem_dot.set_data([ti], [float(d["T_em"][i])])

        return (
            phasor, trail, hex_sec_txt, hex_time_txt,
            duty_cur, curr_cur, spd_cur, tem_cur,
            spd_dot, tem_dot,
            *duty_dots, *curr_dots, *sec_patches,
            *[inv[f"Q{ph}_{r}"] for ph in "abc" for r in "HL"],
            *[inv[f"T{ph}_{r}"] for ph in "abc" for r in "HL"],
            *[inv[f"I_label_{ph}"]    for ph in "abc"],
            *[inv[f"duty_label_{ph}"] for ph in "abc"],
            inv["sector_badge"], inv["vdc_label"],
        )

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=interval_ms, blit=True,
    )

    out_path = path
    if save_mp4:
        out_path = path.replace(".gif", ".mp4")
        writer   = animation.FFMpegWriter(fps=1000 // interval_ms, bitrate=1400)
        print(f"[Anim] Encoding {n_frames} frames → {out_path}  (ffmpeg) …", flush=True)
        ani.save(out_path, writer=writer, dpi=80)
    else:
        print(f"[Anim] Encoding {n_frames} frames → {out_path}  (pillow GIF) …", flush=True)
        writer_gif = animation.PillowWriter(fps=1000 // interval_ms)
        writer_gif.setup(fig, out_path, dpi=80)
        for k in range(n_frames):
            _update(k)
            writer_gif.grab_frame()
            if k % 20 == 0 or k == n_frames - 1:
                print(f"[Anim]   frame {k + 1:3d}/{n_frames}", flush=True)
        writer_gif.finish()

    plt.close(fig)
    print(f"[Plot] {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 68)
    print("  EmbedSim — NANOTEC DB42S02  Open-loop V/f")
    print("  Library  : fs_electrical_machines")
    print("  Blocks   : SpeedRamp → VfAngle → VfDQ + VfTheta → InvPark")
    print("             → SVPWMPack → [cg_start] → SVPWMBlock → [cg_end]")
    print("  Plant    : DB42S02PlantBlock (PMSM_Plant_FMUBlock / PMSM_Plant_FMU.fmu)")
    print("  CodeGen  : Input_T empty  |  Output_T {ta, tb, tc, sector}")
    print("             SVPWMBlock C_CUSTOM_EMIT: SVM_CalculateDutyCycle")
    print(f"  Target   : {OMEGA_CMD_RPM:.0f} RPM  V_dc={V_DC} V  p={P_POLES}  "
          f"dt={DT*1e6:.0f} µs  T={T_SIM} s")
    print(f"  VF_RATIO : {VF_RATIO:.6f} V·s/rad  boost={VF_BOOST:.3f} V  "
          f"T_load={T_LOAD} N·m")
    print("=" * 68)

    data = build_and_run()
    plot_results(data)
    plot_phasor_static(data)
    animate_phasor(data)

    print("\n[Done]")
    print("  db42s02_openloop_results.png")
    print("  db42s02_phasor_sectors.png")
    print("  db42s02_phasor_anim.gif")
    print("  db42s02_topology.html")
    print("  embedsim_gen/EmbedSim_step.c")
    print("  embedsim_gen/EmbedSim_step.h")
