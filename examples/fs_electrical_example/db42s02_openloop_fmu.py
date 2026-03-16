"""
db42s02_openloop_fmu.py
=======================
EmbedSim  —  Open-loop V/f control  —  NANOTEC DB42S02
=======================================================

Library: fs_electrical_machines

Block diagram (Python-first, EmbedSim canonical order):
─────────────────────────────────────────────────────────────────
  CodeGenStart
       │
  SpeedRampBlock            omega_m_ref [rad/s]
       │                    motor_utility_blocks.c → SpeedRamp_Step
  VfAngleBlock              [v_d=0, v_q, theta_e]
       │                    motor_utility_blocks.c → VfAngle_Step
       ├─── VfDQBlock        [v_d, v_q]
       │    motor_utility_blocks.c → VfDQ_Step
       └─── VfThetaBlock     [theta_e]
            motor_utility_blocks.c → VfTheta_Step
                              ↓
  InvParkTransformBlock     [v_alpha, v_beta]
       │                    coordinate_transform.c → InvPark_Step
       ├────────────────────────────────────────► DutyPackBlock (port 0)
       └──► SVPWMBlock       [T1, T2, T0, sector]
                │            svpwm.c → SVPWM_Step
                └────────────────────────────── DutyPackBlock (port 1)
  DutyPackBlock             [duty_a, duty_b, duty_c, V_dc, 0]
       │                    motor_utility_blocks.c → DutyPack_Step
  CodeGenEnd                → cg_end.generate_loop() → LoopGenerator
       │
  FMUSinkBlock              PMSM plant (NOT code-generated)
─────────────────────────────────────────────────────────────────

CodeGen strategy
────────────────
  Every block carries PYX_FILE.
  PYXInspector auto-populates step_func / state_struct / init_func
  at class-definition time via VectorBlock.__init_subclass__.
  LoopGenerator emits typed C calls for every block.
  Zero C_CUSTOM_EMIT.  Zero hand-written C strings in this file.

Motor: NANOTEC DB42S02  (PMSM_Motor.mo)
  R=0.19 Ω  Ld=Lq=0.125 mH  λ_pm=0.0014 Wb  J=2.4e-6 kg·m²
  p=4  V_dc=17 V

Build wrappers once:
    cd fs_electrical_machines/c_src
    python setup_motor_utility_blocks.py build_ext --inplace

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
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE
for _ in range(6):
    if (_ROOT / "embedsim").is_dir():
        break
    _ROOT = _ROOT.parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_FS_ELEC = _ROOT / "fs_electrical_machines"
if str(_FS_ELEC) not in sys.path:
    sys.path.insert(0, str(_FS_ELEC))

# ── EmbedSim — full engine ────────────────────────────────────────────────────
from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.code_generator import CodeGenStart, CodeGenEnd

import codegen_db42s02  # attaches C_CUSTOM_EMIT to the three non-standard blocks

# ── fs_electrical_machines — all blocks have compiled .pyd wrappers ────────────
from motor_utility_blocks import (
    SpeedRampBlock,
    VfAngleBlock,
    VfDQBlock,
    VfThetaBlock,
    DutyPackBlock,
    SVPWMPackBlock,
)
from coordinate_transform_blocks import InvParkTransformBlock
from svpwm_block                 import SVPWMBlock
from PMSM_MotorBlock             import PMSM_MotorBlock

_FMU_PATH = str(_FS_ELEC / "modelica" / "PMSM_Motor.fmu")


# =============================================================================
# Motor constants  (PMSM_Motor.mo)
# =============================================================================
P_POLES        = 4
V_DC           = 17.0
V_PHASE_PEAK   = V_DC / math.sqrt(3.0)
OMEGA_M_RATED  = 8000.0 * 2.0 * math.pi / 60.0
OMEGA_E_RATED  = P_POLES * OMEGA_M_RATED
VF_RATIO       = V_PHASE_PEAK / OMEGA_E_RATED

T_SIM          = 0.4
DT             = 1e-4
OMEGA_CMD_RPM  = 400.0
OMEGA_CMD_RADS = OMEGA_CMD_RPM * 2.0 * math.pi / 60.0
RAMP_TIME      = 0.15


# =============================================================================
# FMUSinkBlock  — simulation plant, NOT code-generated
# =============================================================================
class FMUSinkBlock(VectorBlock):
    """
    Wraps PMSM_MotorBlock FMU from fs_electrical_machines.
    Analytical Euler dq fallback when FMU is unavailable.
    Outside CodeGenStart/CodeGenEnd — never touched by LoopGenerator.
    """
    # Topology printer uses these class attributes for categorisation
    TOPO_CATEGORY = "plant"
    output_label  = "[rpm,ia,ib,ic,Tem]"

    def __init__(self, name: str, fmu_path: str) -> None:
        super().__init__(name)
        self.is_dynamic = False
        try:
            self._motor   = PMSM_MotorBlock(name="pmsm", fmu_path=fmu_path)
            self._has_fmu = True
            print(f"[FMU] Loaded: {fmu_path}")
        except Exception as exc:
            print(f"[FMU] Not available ({exc}) — using analytical fallback")
            self._has_fmu = False
            self._motor   = None
            self._id = self._iq = self._omega_m = self._th = 0.0

        self.speed_rpm = self.i_a = self.i_b = self.i_c = 0.0
        self.theta_e_motor = self.T_em = 0.0

    def compute_py(self, t, dt, input_values=None):
        da = db = dc = 0.5
        vdc = V_DC
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 4:
                da, db, dc, vdc = (float(v[0]), float(v[1]),
                                   float(v[2]), float(v[3]))

        if self._has_fmu:
            sig = VectorSignal(
                np.array([da, db, dc, vdc, 0.0], dtype=DEFAULT_DTYPE))
            self._motor.compute_py(t, dt, [sig])
            self.speed_rpm     = self._motor.read_speed_rpm()
            self.i_a           = self._motor.read_i_a()
            self.i_b           = self._motor.read_i_b()
            self.i_c           = self._motor.read_i_c()
            self.theta_e_motor = self._motor.read_theta_e()
            self.T_em          = self._motor.read_T_em()
        else:
            R = 0.19; Ld = Lq = 0.125e-3; lam = 0.0014
            Jm = 2.4e-6; Bf = 1e-6; H = math.sqrt(3.0) / 2.0
            van = da*vdc; vbn = db*vdc; vcn = dc*vdc
            vn  = (van + vbn + vcn) / 3.0
            va  = van - vn; vb = vbn - vn; vc = vcn - vn
            va_a = (2.0/3.0)*(va - 0.5*vb - 0.5*vc)
            vb_a = (2.0/3.0)*(H*vb - H*vc)
            th   = self._th
            vd   =  va_a*math.cos(th) + vb_a*math.sin(th)
            vq   = -va_a*math.sin(th) + vb_a*math.cos(th)
            oe   = P_POLES * self._omega_m
            self._id += (vd - R*self._id + oe*Lq*self._iq) / Ld * dt
            self._iq += (vq - R*self._iq - oe*(Ld*self._id + lam)) / Lq * dt
            Tem  = 1.5*P_POLES*(lam*self._iq + (Ld-Lq)*self._id*self._iq)
            self._omega_m += (Tem - Bf*self._omega_m) / Jm * dt
            self._th       = math.fmod(self._th + oe*dt, 2.0*math.pi)
            if self._th < 0.0:
                self._th += 2.0*math.pi
            ia = self._id*math.cos(th) - self._iq*math.sin(th)
            ib = self._id*math.sin(th) + self._iq*math.cos(th)
            self.i_a           =  ia
            self.i_b           = -0.5*ia + H*ib
            self.i_c           = -0.5*ia - H*ib
            self.speed_rpm     = self._omega_m * 60.0 / (2.0*math.pi)
            self.theta_e_motor = self._th
            self.T_em          = Tem

        self.output = VectorSignal(
            np.array([self.speed_rpm, self.i_a, self.i_b,
                      self.i_c, self.T_em], dtype=DEFAULT_DTYPE),
            self.name)
        return self.output


# =============================================================================
# Build & run
# =============================================================================
def build_and_run() -> dict:
    """
    Wire all blocks, register signals on sim.scope, hand to EmbedSim
    for topology sort + time loop, export topology HTML via sim.topo,
    then call LoopGenerator.

    Signal recording : sim.scope.add(block, indices, label) before sim.run()
    Signal retrieval : sim.scope.get_signal(label, index) after sim.run()
    No VectorEnd sinks for data.  No hand-rolled loop.  No _h() hack.
    """

    # ── Instantiate ──────────────────────────────────────────────────────────
    cg_start  = CodeGenStart("cg_start")
    speed_ref = SpeedRampBlock("speed_ref",
                               omega_target=OMEGA_CMD_RADS,
                               ramp_time=RAMP_TIME)
    vf_angle  = VfAngleBlock("vf_angle",
                             vf_ratio=VF_RATIO,
                             v_phase_peak=V_PHASE_PEAK,
                             p_poles=P_POLES)
    vf_dq     = VfDQBlock("vf_dq")
    vf_theta  = VfThetaBlock("vf_theta")
    inv_park  = InvParkTransformBlock("inv_park", use_c_backend=False)
    svpwm_pack = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)   # polar adapter
    svpwm     = SVPWMBlock("svpwm", use_c_backend=False)
    duty_pack = DutyPackBlock("duty_pack", v_dc=V_DC)
    cg_end    = CodeGenEnd("cg_end")
    motor     = FMUSinkBlock("motor_sink", fmu_path=_FMU_PATH)
    sink      = VectorEnd("sink")      # terminal — main path
    sink_cg   = VectorEnd("sink_cg")   # terminal — CodeGen boundary branches

    # ── Wire ─────────────────────────────────────────────────────────────────
    # Data path — controller chain
    speed_ref  >> vf_angle
    vf_angle   >> vf_dq
    vf_angle   >> vf_theta
    vf_dq      >> inv_park             # port 0: [v_d, v_q]
    vf_theta   >> inv_park             # port 1: [theta_e]
    inv_park   >> svpwm_pack           # [v_alpha, v_beta] → polar conversion
    svpwm_pack >> svpwm                # [Vref, alpha, Vdc] → sector detection
    inv_park   >> duty_pack            # port 0: [v_alpha, v_beta]
    svpwm      >> duty_pack            # port 1: [T1, T2, T0, sector]
    duty_pack  >> motor
    motor      >> sink                 # terminal sink — main DFS path

    # CodeGen boundary markers — parallel branches off the data path.
    # Both feed a dedicated sink_cg so the DFS reaches them and they
    # appear in the topology with the dashed CodeGen region highlight.
    speed_ref  >> cg_start             # marks controller region input
    duty_pack  >> cg_end               # marks controller region output
    cg_start   >> sink_cg             # keeps cg_start in the DFS graph
    cg_end     >> sink_cg             # keeps cg_end   in the DFS graph

    # ── EmbedSim ─────────────────────────────────────────────────────────────
    sim = EmbedSim(
        sinks  = [sink, sink_cg],
        T      = T_SIM,
        dt     = DT,
        solver = ODESolver.EULER,
    )

    # ── Register signals on scope (MUST be before sim.run()) ─────────────────
    sim.scope.add(speed_ref,   indices=[0],             label="omega_ref")
    sim.scope.add(vf_angle,    indices=[0, 1, 2],       label="vf_angle")
    sim.scope.add(inv_park,    indices=[0, 1],           label="inv_park")
    sim.scope.add(svpwm_pack,  indices=[0, 1, 2],       label="svpwm_pack")
    sim.scope.add(svpwm,       indices=[0, 1, 2, 3],    label="svpwm")
    sim.scope.add(duty_pack,   indices=[0, 1, 2],       label="duty_pack")
    sim.scope.add(motor,       indices=[0, 1, 2, 3, 4], label="motor")

    # ── Topology: console ASCII + interactive HTML ────────────────────────────
    # wire_labels maps (src_name, dst_name) → signal label shown on the arrow.
    # Every edge in the diagram carries the signal name / dimension.
    _wire_labels = {
        ("speed_ref",  "cg_start"):   "ω_ref [rad/s]",
        ("speed_ref",  "vf_angle"):   "ω_ref [rad/s]",
        ("vf_angle",   "vf_dq"):      "[v_d, v_q, θ_e]",
        ("vf_angle",   "vf_theta"):   "[v_d, v_q, θ_e]",
        ("vf_dq",      "inv_park"):   "[v_d, v_q]",
        ("vf_theta",   "inv_park"):   "θ_e",
        ("inv_park",   "svpwm_pack"): "[v_α, v_β]",
        ("svpwm_pack", "svpwm"):      "[Vref, α, Vdc]",
        ("inv_park",   "duty_pack"):  "[v_α, v_β]",
        ("svpwm",      "duty_pack"):  "[T1, T2, T0, sec]",
        ("duty_pack",  "cg_end"):     "[da, db, dc, Vdc, 0]",
        ("duty_pack",  "motor_sink"): "[da, db, dc, Vdc, 0]",
        ("cg_start",   "sink_cg"):    "ω_ref [rad/s]",
        ("cg_end",     "sink_cg"):    "[da, db, dc]",
        ("motor_sink", "sink"):       "[rpm, ia, ib, ic, Tem]",
    }

    print("\n[Topology] Signal-flow diagram:")
    sim.topo.print_console()
    _topo_path = str(_HERE / "db42s02_topology.html")
    sim.topo.export_html(_topo_path, wire_labels=_wire_labels)
    print(f"[Topology] {_topo_path}")

    # ── Run ───────────────────────────────────────────────────────────────────
    sim.run()

    # ── Extract signals from scope ────────────────────────────────────────────
    sc = sim.scope
    hist = {
        "t":         np.array(sc.t, dtype=np.float32),
        # omega_ref: scope records rad/s → convert to RPM
        "omega_ref": sc.get_signal("omega_ref", 0) * 60.0 / (2.0 * math.pi),
        "v_d":       sc.get_signal("vf_angle",  0),
        "v_q":       sc.get_signal("vf_angle",  1),
        "theta_e":   sc.get_signal("vf_angle",  2),
        "v_alpha":   sc.get_signal("inv_park",  0),
        "v_beta":    sc.get_signal("inv_park",  1),
        "sector":    sc.get_signal("svpwm",     3),
        "duty_a":    sc.get_signal("duty_pack", 0),
        "duty_b":    sc.get_signal("duty_pack", 1),
        "duty_c":    sc.get_signal("duty_pack", 2),
        "speed_rpm": sc.get_signal("motor",     0),
        "i_a":       sc.get_signal("motor",     1),
        "i_b":       sc.get_signal("motor",     2),
        "i_c":       sc.get_signal("motor",     3),
        "T_em":      sc.get_signal("motor",     4),
    }

    # ── LoopGenerator — embedsim_loop.c / .h ─────────────────────────────────
    print("\n[CodeGen] Calling cg_end.generate_loop() …")
    cg_end.generate_loop(
        cg_start=cg_start,
        output_dir=_ROOT,
        dt_hz=1.0 / DT,
        write_files=True,
    )

    # ── StepGenerator — <Prefix>_step.c / .h ─────────────────────────────────
    # Reads cg_start.iter_signals() → EmbedSim_Input_T
    # Reads cg_end.iter_signals()   → EmbedSim_Output_T
    # Emits EmbedSim_Step(dt, in*, out*) — Simulink-equivalent, fully general.
    print("\n[CodeGen] Calling cg_end.generate_step() …")
    cg_end.generate_step(
        cg_start   = cg_start,
        output_dir = _ROOT,
        dt_hz      = 1.0 / DT,
        prefix     = "EmbedSim",
        write_files = True,
    )

    return hist


# =============================================================================
# Static plots
# =============================================================================
_SECTOR_COLORS = ["#FF595E", "#FF924C", "#FFCA3A",
                  "#8AC926", "#1982C4", "#6A4C93"]


def plot_results(d: dict, path: str = "db42s02_openloop_results.png"):
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        f"NANOTEC DB42S02 — Open-loop V/f  "
        f"(cmd {OMEGA_CMD_RPM:.0f} RPM | V_dc {V_DC} V | p={P_POLES})",
        fontsize=13, fontweight="bold")
    t = d["t"]

    axes[0].plot(t, d["omega_ref"], "k--", lw=1.2, label="ω_ref [RPM]")
    axes[0].plot(t, d["speed_rpm"], "C0",  lw=1.5, label="ω_motor FMU [RPM]")
    axes[0].set_ylabel("Speed [RPM]"); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Motor speed  (fs_electrical_machines FMU plant)")

    axes[1].plot(t, d["i_a"], "C3", lw=0.9, label="i_a")
    axes[1].plot(t, d["i_b"], "C2", lw=0.9, label="i_b")
    axes[1].plot(t, d["i_c"], "C1", lw=0.9, label="i_c")
    axes[1].set_ylabel("Current [A]"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3); axes[1].set_title("Phase currents — FMU sensor output")

    axes[2].plot(t, d["duty_a"], "C3", lw=0.8, label="duty_a → GTM TOM0 CH0")
    axes[2].plot(t, d["duty_b"], "C2", lw=0.8, label="duty_b → GTM TOM0 CH2")
    axes[2].plot(t, d["duty_c"], "C1", lw=0.8, label="duty_c → GTM TOM0 CH4")
    axes[2].set_ylabel("Duty [0–1]"); axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3); axes[2].set_ylim(0, 1)
    axes[2].set_title("PWM duty cycles  (DutyPackBlock → AURIX GTM TOM)")

    ax4 = axes[3]; ax4b = ax4.twinx()
    ax4.step(t, d["sector"], where="post", color="C5", lw=1.2, label="SVPWM sector")
    ax4b.plot(t, d["v_q"], "C0--", lw=1.0, label="v_q [V]")
    ax4.set_ylabel("Sector [1–6]", color="C5"); ax4b.set_ylabel("v_q [V]", color="C0")
    ax4.set_ylim(0, 7); ax4.set_yticks([1, 2, 3, 4, 5, 6])
    ax4.set_xlabel("Time [s]"); ax4.grid(alpha=0.3)
    ax4.set_title("SVPWM sector + v_q")
    l1, n1 = ax4.get_legend_handles_labels(); l2, n2 = ax4b.get_legend_handles_labels()
    ax4.legend(l1+l2, n1+n2, fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Plot] {path}")


def plot_phasor_static(d: dict, path: str = "db42s02_phasor_sectors.png"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_title(
        f"SVPWM hexagon — α-β plane\n"
        f"NANOTEC DB42S02  {OMEGA_CMD_RPM:.0f} RPM open-loop V/f",
        fontsize=12, fontweight="bold")
    R = 2.0 / 3.0
    for s in range(6):
        a0 = math.radians(60*s); a1 = math.radians(60*(s+1))
        angs = np.linspace(a0, a1, 30)
        ax.fill(np.concatenate([[0], R*np.cos(angs), [0]]),
                np.concatenate([[0], R*np.sin(angs), [0]]),
                color=_SECTOR_COLORS[s], alpha=0.22)
        am = (a0+a1)/2.0
        ax.text(0.45*math.cos(am), 0.45*math.sin(am), f"S{s+1}",
                ha="center", va="center", fontsize=13,
                fontweight="bold", color=_SECTOR_COLORS[s])
    ax.plot([R*math.cos(k*math.pi/3) for k in range(7)],
            [R*math.sin(k*math.pi/3) for k in range(7)], "k-", lw=1.5)
    for k in range(6):
        ax.text(R*math.cos(k*math.pi/3)*1.08, R*math.sin(k*math.pi/3)*1.08,
                f"V{k+1}", ha="center", fontsize=9, fontweight="bold")
    sc = 1.0 / V_DC
    ax.plot(d["v_alpha"]*sc, d["v_beta"]*sc,
            "navy", lw=0.5, alpha=0.35, label="α-β trajectory")
    vx = float(d["v_alpha"][-1])*sc; vy = float(d["v_beta"][-1])*sc
    ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=2.5))
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("α  (normalised to V_dc)"); ax.set_ylabel("β")
    ax.set_xlim(-0.85, 0.85); ax.set_ylim(-0.85, 0.85); ax.grid(alpha=0.25)
    ax.legend(handles=[mpatches.Patch(color=_SECTOR_COLORS[i], alpha=0.5,
              label=f"Sector {i+1}") for i in range(6)],
              fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Plot] {path}")


# =============================================================================
# FOC six-sector animated phasor hexagon  +  phase current scrolling plot
# =============================================================================
def animate_phasor_hexagon(d: dict,
                           path: str = "db42s02_phasor_anim.gif",
                           n_frames: int = 150,
                           fps: int = 25) -> None:
    """
    Three-panel animated GIF.

    Left        : FOC six-sector Hehagen hexagon.  The voltage phasor is
                  drawn as a proper FancyArrow (shaft + arrowhead) from the
                  origin, colored by active sector.  Trajectory trace fades
                  behind it.
    Right-top   : Scrolling phase current waveforms i_a / i_b / i_c with
                  moving cursor and live readout.
    Right-bottom: PWM duty cycle bar gauges for duty_a / duty_b / duty_c.
                  Each phase is a vertical bar that fills 0→1, matching the
                  DutyPackBlock output that drives the AURIX GTM TOM channels.
    """
    print(f"[Anim] Building {n_frames}-frame three-panel animation …")

    # ── Decimate data ─────────────────────────────────────────────────────────
    idx     = np.linspace(0, len(d["t"]) - 1, n_frames, dtype=int)
    t_dec   = d["t"][idx]
    # Normalise by V_PHASE_PEAK (= V_dc/sqrt(3)), NOT V_dc.
    # The SVPWM hexagon boundary is at |V_ref| = V_phase_peak,
    # so R=2/3 in this normalisation fills ~2/3 of the hexagon
    # at rated voltage — correct at any operating speed.
    # Negate v_beta: InvPark emits v_beta = v_d*sin+v_q*cos;
    # display convention wants CCW rotation for positive omega_e.
    # Dynamic normalisation: scale so the steady-state phasor fills
    # ~60% of the hexagon radius regardless of operating speed.
    # This keeps the arrow visible at 400 RPM (5% of rated voltage)
    # while still showing the correct circular trajectory shape.
    _v_mag_peak = float(np.max(np.sqrt(
        d["v_alpha"]**2 + d["v_beta"]**2))) + 1e-9
    _VNORM  = _v_mag_peak / 0.60    # map peak magnitude to r=0.60
    va_dec  =  d["v_alpha"][idx] / _VNORM
    vb_dec  = -d["v_beta"][idx]  / _VNORM
    sec_dec = d["sector"][idx].astype(int)
    rpm_dec = d["speed_rpm"][idx]
    vq_dec  = d["v_q"][idx]
    ia_dec  = d["i_a"][idx]
    ib_dec  = d["i_b"][idx]
    ic_dec  = d["i_c"][idx]
    da_dec  = d["duty_a"][idx]
    db_dec  = d["duty_b"][idx]
    dc_dec  = d["duty_c"][idx]

    t_full  = d["t"]
    ia_full = d["i_a"]
    ib_full = d["i_b"]
    ic_full = d["i_c"]

    WIN    = 0.04
    i_peak = max(np.max(np.abs(ia_full)),
                 np.max(np.abs(ib_full)),
                 np.max(np.abs(ic_full))) * 1.25
    if i_peak < 0.01:
        i_peak = 0.5

    # ── Figure: 1 left column + 2 right rows ─────────────────────────────────
    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle(
        f"NANOTEC DB42S02 — Open-loop V/f  "
        f"{OMEGA_CMD_RPM:.0f} RPM  V_dc={V_DC} V  p={P_POLES}",
        fontsize=12, fontweight="bold", color="white")

    gs = fig.add_gridspec(2, 2,
                          width_ratios=[1, 1.2],
                          height_ratios=[1.4, 1],
                          hspace=0.38, wspace=0.32,
                          left=0.06, right=0.97,
                          top=0.91, bottom=0.08)

    ax_hex = fig.add_subplot(gs[:, 0])   # left: full height
    ax_cur = fig.add_subplot(gs[0, 1])   # right-top: currents
    ax_pwm = fig.add_subplot(gs[1, 1])   # right-bottom: duty bars

    for _ax in (ax_hex, ax_cur, ax_pwm):
        _ax.set_facecolor("#181818")
        _ax.tick_params(colors="lightgray", labelsize=8)
        for spine in _ax.spines.values():
            spine.set_edgecolor("#444444")

    # ── LEFT: hexagon ─────────────────────────────────────────────────────────
    ax_hex.set_aspect("equal")
    ax_hex.set_xlim(-0.85, 0.85)
    ax_hex.set_ylim(-0.85, 0.85)
    ax_hex.axhline(0, color="#444444", lw=0.5)
    ax_hex.axvline(0, color="#444444", lw=0.5)
    ax_hex.grid(alpha=0.15, color="#555555")
    ax_hex.set_xlabel("α  (norm. to Vₚₕₐₛₑ)", fontsize=9, color="lightgray")
    ax_hex.set_ylabel("β", fontsize=9, color="lightgray")
    ax_hex.set_title("α-β voltage phasor  (SVPWM sectors)",
                     fontsize=10, color="white", pad=6)

    R = 2.0 / 3.0

    # ── Static sector wedges — dim background only, NO live highlight ─────────
    # Standard SVPWM convention: S1 centred on +alpha (0 deg), boundaries at +/-30 deg.
    for s in range(6):
        a0   = math.radians(60 * s - 30)
        a1   = math.radians(60 * s + 30)
        angs = np.linspace(a0, a1, 30)
        xs   = np.concatenate([[0], R * np.cos(angs), [0]])
        ys   = np.concatenate([[0], R * np.sin(angs), [0]])
        ax_hex.fill(xs, ys, color=_SECTOR_COLORS[s], alpha=0.08, zorder=1)
        am = math.radians(60 * s)
        ax_hex.text(0.48 * math.cos(am), 0.48 * math.sin(am), f"S{s+1}",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color=_SECTOR_COLORS[s], alpha=0.55)

    ax_hex.plot(
        [R * math.cos(k * math.pi / 3) for k in range(7)],
        [R * math.sin(k * math.pi / 3) for k in range(7)],
        color="#cccccc", lw=1.2, zorder=3)
    for k in range(6):
        ax_hex.text(
            R * math.cos(k * math.pi / 3) * 1.10,
            R * math.sin(k * math.pi / 3) * 1.10,
            f"V{k+1}", ha="center", va="center",
            fontsize=7, fontweight="bold", color="#aaaaaa")

    # ── Phase-axis reference dashes  (0 deg, -120 deg, +120 deg  = a, b, c) ──
    # Teaches: each phase has a fixed spatial axis in alpha-beta space.
    _PHASE_ANGLES = [0.0, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0]
    _PHASE_COLORS = ["#ff4444", "#44cc44", "#4488ff"]
    _PHASE_LABELS = ["a", "b", "c"]
    _SPOKE_R      = 0.78
    for ang, col, lbl in zip(_PHASE_ANGLES, _PHASE_COLORS, _PHASE_LABELS):
        ax_hex.plot([0, _SPOKE_R * math.cos(ang)],
                    [0, _SPOKE_R * math.sin(ang)],
                    color=col, lw=0.8, ls="--", alpha=0.35, zorder=2)
        ax_hex.text(_SPOKE_R * 1.06 * math.cos(ang),
                    _SPOKE_R * 1.06 * math.sin(ang),
                    lbl, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=col, alpha=0.7)

    # ── Radial current spokes — live, one per phase ───────────────────────────
    # Each spoke runs from origin along the phase axis.
    # Length = i_x / i_peak * _SPOKE_R  (signed: pos -> along +axis, neg -> opposite).
    # A filled dot marks the current tip, showing the student which phases
    # are conducting and how the resultant (phasor) is synthesised.
    _spoke_lines = []
    _spoke_dots  = []
    for col in _PHASE_COLORS:
        ln, = ax_hex.plot([], [], color=col, lw=2.5, solid_capstyle="round",
                          zorder=7)
        dt, = ax_hex.plot([], [], "o", color=col, ms=7, zorder=8)
        _spoke_lines.append(ln)
        _spoke_dots.append(dt)

    traj_line, = ax_hex.plot([], [], color="#3399ff", lw=0.7,
                              alpha=0.30, zorder=2)

    # FancyArrow: drawn fresh each frame, stored in a list so we can remove it
    _arrow_container = [None]

    info_box = ax_hex.text(
        0.02, 0.98, "", transform=ax_hex.transAxes,
        fontsize=9, va="top", ha="left", color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="#222222", alpha=0.85),
        fontfamily="monospace", zorder=9)
    ax_hex.legend(
        handles=[mpatches.Patch(color=c, alpha=0.7, label=f"i_{l}")
                 for c, l in zip(_PHASE_COLORS, _PHASE_LABELS)],
        fontsize=8, loc="lower right", framealpha=0.5,
        facecolor="#222222", labelcolor="white")

    # ── RIGHT-TOP: phase currents ─────────────────────────────────────────────
    ax_cur.set_xlim(0.0, WIN)
    ax_cur.set_ylim(-i_peak, i_peak)
    ax_cur.set_xlabel("Time in window [ms]", fontsize=9, color="lightgray")
    ax_cur.set_ylabel("Current [A]", fontsize=9, color="lightgray")
    ax_cur.set_title("Phase currents  i_a / i_b / i_c  (FMU sensor)",
                     fontsize=10, color="white", pad=4)
    ax_cur.axhline(0, color="#444444", lw=0.6)
    ax_cur.grid(alpha=0.15, color="#555555")
    ax_cur.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x*1000:.0f}"))

    ax_cur.plot(t_full, ia_full, color="#ff6666", lw=0.3, alpha=0.10)
    ax_cur.plot(t_full, ib_full, color="#66ff66", lw=0.3, alpha=0.10)
    ax_cur.plot(t_full, ic_full, color="#6699ff", lw=0.3, alpha=0.10)

    line_ia, = ax_cur.plot([], [], color="#ff4444", lw=1.4, label="i_a", zorder=4)
    line_ib, = ax_cur.plot([], [], color="#44cc44", lw=1.4, label="i_b", zorder=4)
    line_ic, = ax_cur.plot([], [], color="#4488ff", lw=1.4, label="i_c", zorder=4)
    cursor_line = ax_cur.axvline(0.0, color="white", lw=1.0, alpha=0.7, zorder=5)

    cur_box = ax_cur.text(
        0.98, 0.98, "", transform=ax_cur.transAxes,
        fontsize=8, va="top", ha="right", color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="#222222", alpha=0.85),
        fontfamily="monospace", zorder=7)
    ax_cur.legend(fontsize=7, loc="lower right",
                  framealpha=0.5, facecolor="#222222", labelcolor="white")

    # ── RIGHT-BOTTOM: PWM duty bar gauges ────────────────────────────────────
    # Three vertical bars: A (red), B (green), C (blue)
    # x centres: 0.2, 0.5, 0.8 in axes data coords
    _BAR_X      = [0.2, 0.5, 0.8]
    _BAR_W      = 0.18
    _BAR_COLORS = ["#ff4444", "#44cc44", "#4488ff"]
    _BAR_LABELS = ["duty_a\nTOM0 CH0", "duty_b\nTOM0 CH2", "duty_c\nTOM0 CH4"]

    ax_pwm.set_xlim(0.0, 1.0)
    ax_pwm.set_ylim(0.0, 1.0)
    ax_pwm.set_xticks(_BAR_X)
    ax_pwm.set_xticklabels(_BAR_LABELS, fontsize=8, color="lightgray")
    ax_pwm.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_pwm.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"],
                            fontsize=8, color="lightgray")
    ax_pwm.set_title("PWM duty cycles  (DutyPackBlock → AURIX GTM)",
                     fontsize=10, color="white", pad=4)
    ax_pwm.grid(axis="y", alpha=0.15, color="#555555")
    ax_pwm.axhline(0.5, color="#666666", lw=0.8, ls="--")   # 50 % reference

    # Background slot outlines
    for bx in _BAR_X:
        ax_pwm.add_patch(plt.Rectangle(
            (bx - _BAR_W / 2, 0), _BAR_W, 1.0,
            fc="#222222", ec="#444444", lw=0.8, zorder=1))

    # Live fill bars — start at 0 height
    duty_bars = []
    duty_texts = []
    for bx, bc in zip(_BAR_X, _BAR_COLORS):
        bar = plt.Rectangle(
            (bx - _BAR_W / 2, 0), _BAR_W, 0.0,
            fc=bc, alpha=0.85, zorder=2)
        ax_pwm.add_patch(bar)
        duty_bars.append(bar)
        txt = ax_pwm.text(bx, 0.02, "0.00", ha="center", va="bottom",
                          fontsize=9, color="white", fontweight="bold",
                          fontfamily="monospace", zorder=3)
        duty_texts.append(txt)

    # ── Init ──────────────────────────────────────────────────────────────────
    def _init():
        traj_line.set_data([], [])
        info_box.set_text("")
        line_ia.set_data([], [])
        line_ib.set_data([], [])
        line_ic.set_data([], [])
        cur_box.set_text("")
        for bar in duty_bars:
            bar.set_height(0.0)
        for txt in duty_texts:
            txt.set_text("0.00")
        for ln in _spoke_lines:
            ln.set_data([], [])
        for dt in _spoke_dots:
            dt.set_data([], [])
        return (traj_line, info_box,
                *_spoke_lines, *_spoke_dots,
                line_ia, line_ib, line_ic,
                cur_box, *duty_bars, *duty_texts)

    # ── Update ────────────────────────────────────────────────────────────────
    def _update(frame):
        tf  = float(t_dec[frame])
        vx  = float(va_dec[frame])
        vy  = float(vb_dec[frame])
        sec = int(sec_dec[frame])
        rpm = float(rpm_dec[frame])
        vq  = float(vq_dec[frame])
        ia  = float(ia_dec[frame])
        ib  = float(ib_dec[frame])
        ic  = float(ic_dec[frame])
        da  = float(da_dec[frame])
        db  = float(db_dec[frame])
        dc  = float(dc_dec[frame])

        # ── Hexagon panel ─────────────────────────────────────────────────────
        traj_line.set_data(va_dec[:frame + 1], vb_dec[:frame + 1])

        # Remove previous arrow and draw a new FancyArrow each frame
        if _arrow_container[0] is not None:
            _arrow_container[0].remove()

        if 1 <= sec <= 6:
            arrow_color = _SECTOR_COLORS[sec - 1]
        else:
            arrow_color = "#ff4444"

        mag = math.sqrt(vx * vx + vy * vy)
        if mag > 1e-4:
            # Shaft length slightly shorter than full mag so head fits cleanly
            hw = 0.055          # head width
            hl = 0.07           # head length
            shaft_x = vx * (1.0 - hl / mag)
            shaft_y = vy * (1.0 - hl / mag)
            arrow = mpatches.FancyArrow(
                0, 0, shaft_x, shaft_y,
                width=0.012,
                head_width=hw,
                head_length=hl,
                length_includes_head=False,
                color=arrow_color,
                zorder=6)
        else:
            arrow = mpatches.FancyArrow(
                0, 0, 0, 0,
                width=0.001, head_width=0.001, head_length=0.001,
                color=arrow_color, zorder=6)

        ax_hex.add_patch(arrow)
        _arrow_container[0] = arrow

        # ── Radial current spokes ─────────────────────────────────────
        # Project each phase current onto its fixed αβ axis:
        #   a-axis: 0 deg    b-axis: -120 deg    c-axis: +120 deg
        # The spoke length is i_x / i_peak * _SPOKE_R (signed).
        # Positive current → tip along +axis; negative → tip reversed.
        # The resultant of the three signed spokes equals the voltage
        # phasor direction — the core insight of space-vector theory.
        for ln, dt, ang, i_val in zip(
                _spoke_lines, _spoke_dots, _PHASE_ANGLES, [ia, ib, ic]):
            scale = (i_val / i_peak) * _SPOKE_R if i_peak > 1e-6 else 0.0
            ex = scale * math.cos(ang)
            ey = scale * math.sin(ang)
            ln.set_data([0, ex], [0, ey])
            dt.set_data([ex], [ey])

        info_box.set_text(
            f"t  = {tf*1000:6.1f} ms\n"
            f"ω  = {rpm:7.1f} RPM\n"
            f"Vq = {vq:7.3f} V\n"
            f"S  =     {sec}")

        # ── Current panel ─────────────────────────────────────────────────────
        t_win_start = max(0.0, tf - WIN * 0.75)
        t_win_end   = t_win_start + WIN
        ax_cur.set_xlim(t_win_start, t_win_end)
        mask = (t_full >= t_win_start) & (t_full <= t_win_end)
        t_w  = t_full[mask]
        line_ia.set_data(t_w, ia_full[mask])
        line_ib.set_data(t_w, ib_full[mask])
        line_ic.set_data(t_w, ic_full[mask])
        cursor_line.set_xdata([tf, tf])
        ax_cur.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x*1000:.0f}"))
        cur_box.set_text(
            f"i_a = {ia:+.3f} A\n"
            f"i_b = {ib:+.3f} A\n"
            f"i_c = {ic:+.3f} A")

        # ── PWM duty bar gauges ───────────────────────────────────────────────
        for bar, val, txt in zip(duty_bars, [da, db, dc], duty_texts):
            h = float(np.clip(val, 0.0, 1.0))
            bar.set_height(h)
            txt.set_position((txt.get_position()[0], max(h + 0.02, 0.04)))
            txt.set_text(f"{h:.2f}")

        return (traj_line, info_box,
                *_spoke_lines, *_spoke_dots,
                line_ia, line_ib, line_ic,
                cur_box, cursor_line, *duty_bars, *duty_texts)

    # ── Render ────────────────────────────────────────────────────────────────
    anim_obj = animation.FuncAnimation(
        fig, _update, frames=n_frames,
        init_func=_init, interval=int(1000 / fps), blit=False)
    anim_obj.save(path, writer="pillow", fps=fps, dpi=120)
    plt.close(fig)
    print(f"[Anim] {path}  ({n_frames} frames @ {fps} fps)")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print("=" * 64)
    print("  EmbedSim — NANOTEC DB42S02  Open-loop V/f")
    print("  Library  : fs_electrical_machines")
    print("  Blocks   : motor_utility_blocks.c (SpeedRamp, VfAngle,")
    print("             VfDQ, VfTheta, DutyPack)")
    print("             coordinate_transform.c  (InvPark)")
    print("             svpwm.c                 (SVPWM)")
    print("  CodeGen  : LoopGenerator (feature 05121967)")
    print("             PYX_FILE + PYXInspector on every block")
    print("             Zero C_CUSTOM_EMIT — zero hand-written C strings")
    print("  Topology : TopologyPrinter → db42s02_topology.html")
    print("  Animation: FOC six-sector phasor + phase currents (Hehagen)")
    print(f"  Target   : {OMEGA_CMD_RPM:.0f} RPM  V_dc={V_DC} V  "
          f"p={P_POLES}  dt={DT*1e6:.0f} µs")
    print("=" * 64)

    data = build_and_run()
    plot_results(data)
    plot_phasor_static(data)
    animate_phasor_hexagon(data, n_frames=150, fps=25)

    print("\n[Done]")
    print("  db42s02_openloop_results.png")
    print("  db42s02_phasor_sectors.png")
    print("  db42s02_phasor_anim.gif")
    print("  db42s02_topology.html")
    print("  embedsim_gen/embedsim_loop.c")
    print("  embedsim_gen/embedsim_loop.h")
